import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
import os 
import csv
def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
import os
import csv
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class CMCLIPLoss(ClipLoss):
    """Contrastive Medical CLIP Loss with multi-task learning and result logging."""
    
    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
        output_dir: str = "/root/MedCLIP-SAMv2-main/biomedclip_finetuning/open_clip/src/logs/results",
        temperature: float = 0.07,
        base_temperature: float = 0.1,
        scale_by_temperature: bool = False,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize CMCLIPLoss.
        
        Args:
            local_loss: Whether to use local contrastive loss
            gather_with_grad: Whether to gather features with gradient
            cache_labels: Whether to cache labels
            rank: Process rank
            world_size: Number of processes
            use_horovod: Whether to use Horovod
            output_dir: Directory to save results
            temperature: Temperature for contrastive loss
            base_temperature: Base temperature for supervised contrastive loss
            scale_by_temperature: Whether to scale loss by temperature
            loss_weights: Dictionary of weights for different loss components
        """
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        # Set default loss weights
        default_weights = {
            'contrastive': 1.0,
            'distill': 0.1,
            'location': 10.0,
            'health': 10.0,
            'implicit_category0': 10.0,
            'implicit_category1': 10.0
        }
        self.loss_weights = {**default_weights, **(loss_weights or {})}

        # Initialize supervised contrastive loss
        self.supervised_loss = SupervisedContrastiveLoss(
            temperature=temperature,
            base_temperature=base_temperature,
            scale_by_temperature=scale_by_temperature
        )

        # Initialize projection heads
        input_dim = 512
        self._initialize_projection_heads(input_dim)

        # Setup output directory and results file
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, "classification_results.csv")
        self._setup_output_directory()

    def _initialize_projection_heads(self, input_dim: int):
        """Initialize all projection heads for different tasks."""
        # Location projection heads
        self.image_location_proj = Projection_head(
            input_dim=input_dim, 
            output_dim=input_dim // 4, 
            hidden_dim=2048
        )
        self.text_location_proj = Projection_head(
            input_dim=input_dim, 
            output_dim=input_dim // 4, 
            hidden_dim=2048
        )
        
        # Health projection heads
        self.image_health_proj = Projection_head(
            input_dim=input_dim // 4, 
            output_dim=input_dim // 16, 
            hidden_dim=2048
        )
        self.text_health_proj = Projection_head(
            input_dim=input_dim // 4, 
            output_dim=input_dim // 16, 
            hidden_dim=2048
        )

    def _setup_output_directory(self):
        """Create output directory and initialize results file."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.results_file):
            self._initialize_results_file()

    def _initialize_results_file(self):
        """Initialize CSV file with column headers."""
        headers = [
            'epoch', 'image_name', 
            'contrastive_pred', 
            'location_pred', 'location_true', 'location_pred_name', 'location_true_name',
            'health_pred', 'health_true', 'health_pred_name', 'health_true_name',
            'implicit_category_0', 'implicit_category_1'
        ]
        
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def distillation_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss between teacher and student logits."""
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean()

    def mse_loss(self, teacher_features: torch.Tensor, student_features: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between teacher and student features."""
        return F.mse_loss(teacher_features, student_features)

    def get_classification_predictions(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        logit_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions for contrastive, location, and health tasks.
        
        Returns:
            Tuple of (contrastive_preds, location_preds, health_preds)
        """
        # Contrastive predictions
        logits_per_image, _ = self.get_logits(image_features, text_features, logit_scale)
        contrastive_preds = torch.argmax(logits_per_image, dim=1)
        
        # Location predictions
        text_location_proj = self.text_location_proj(text_features)
        image_location_proj = self.image_location_proj(image_features)
        location_logits = torch.matmul(image_location_proj, text_location_proj.t()) * logit_scale
        location_preds = torch.argmax(location_logits, dim=1)
        
        # Health predictions
        text_health_proj = self.text_health_proj(text_location_proj)
        image_health_proj = self.image_health_proj(image_location_proj)
        health_logits = torch.matmul(image_health_proj, text_health_proj.t()) * logit_scale
        health_preds = torch.argmax(health_logits, dim=1)
        
        return contrastive_preds, location_preds, health_preds

    def get_ground_truth_labels(
        self, 
        location: torch.Tensor, 
        category: torch.Tensor, 
        implicit_category_0: torch.Tensor, 
        implicit_category_1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract ground truth labels from one-hot encodings."""
        location_labels = torch.argmax(location, dim=1)
        category_labels = torch.argmax(category, dim=1)
        implicit_category_0_labels = torch.argmax(implicit_category_0, dim=1)
        implicit_category_1_labels = torch.argmax(implicit_category_1, dim=1)
        
        return location_labels, category_labels, implicit_category_0_labels, implicit_category_1_labels

    def get_classification_results(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        image_names: List[str], 
        location: torch.Tensor, 
        category: torch.Tensor, 
        implicit_category_0: torch.Tensor, 
        implicit_category_1: torch.Tensor, 
        logit_scale: torch.Tensor,
        health_names: List[str], 
        location_names: List[str],
        idx_to_health_label: Dict[int, str], 
        idx_to_location_label: Dict[int, str]
    ) -> List[Dict]:
        """Get classification results for logging."""
        batch_size = image_features.shape[0]
        results = []
        
        # Get predictions
        contrastive_preds, location_preds, health_preds = self.get_classification_predictions(
            image_features, text_features, logit_scale
        )
        
        # Get ground truth labels
        location_labels, category_labels, implicit_category_0_labels, implicit_category_1_labels = \
            self.get_ground_truth_labels(location, category, implicit_category_0, implicit_category_1)
        
        # Compile results for each sample
        for i in range(batch_size):
            # Convert indices to label names
            location_pred_name = idx_to_location_label.get(location_preds[i].item(), "unknown")
            location_true_name = location_names[i] if i < len(location_names) else "unknown"
            
            health_pred_name = idx_to_health_label.get(health_preds[i].item(), "unknown")
            health_true_name = health_names[i] if i < len(health_names) else "unknown"
            
            result = {
                'image_name': image_names[i],
                'contrastive_pred': contrastive_preds[i].item(),
                'location_pred': location_preds[i].item(),
                'location_true': location_labels[i].item(),
                'location_pred_name': location_pred_name,
                'location_true_name': location_true_name,
                'health_pred': health_preds[i].item(),
                'health_true': category_labels[i].item(),
                'health_pred_name': health_pred_name,
                'health_true_name': health_true_name,
                'implicit_category_0': implicit_category_0_labels[i].item(),
                'implicit_category_1': implicit_category_1_labels[i].item()
            }
            results.append(result)
        
        return results

    def save_results_to_csv(self, results: List[Dict], epoch: int):
        """Save classification results to CSV file."""
        with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for result in results:
                writer.writerow([
                    epoch,
                    result['image_name'],
                    result['contrastive_pred'],
                    result['location_pred'],
                    result['location_true'],
                    result['location_pred_name'],
                    result['location_true_name'],
                    result['health_pred'],
                    result['health_true'],
                    result['health_pred_name'],
                    result['health_true_name'],
                    result['implicit_category_0'],
                    result['implicit_category_1']
                ])

    def compute_loss_components(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        teacher_image_features: torch.Tensor,
        teacher_text_features: torch.Tensor,
        location: torch.Tensor,
        category: torch.Tensor,
        implicit_category_0: torch.Tensor,
        implicit_category_1: torch.Tensor,
        epoch: int,
        logit_scale: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute individual loss components."""
        device = image_features.device
        
        # Contrastive loss
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        contrastive_labels = self.get_ground_truth(device, logits_per_image.shape[0])
        contrastive_loss = (
            F.cross_entropy(logits_per_image, contrastive_labels) +
            F.cross_entropy(logits_per_text, contrastive_labels)
        ) / 2

        # Distillation loss
        logits_per_teacher_image, logits_per_teacher_text = self.get_logits(
            teacher_image_features, teacher_text_features, logit_scale
        )
        distill_loss = (
            self.distillation_loss(logits_per_teacher_image, logits_per_image) + 
            self.distillation_loss(logits_per_teacher_text, logits_per_text)
        ) / 2

        # Location loss
        text_location_proj = self.text_location_proj(text_features)
        image_location_proj = self.image_location_proj(image_features)
        location_loss = self.supervised_loss(
            image_location_proj, text_location_proj, location, logit_scale
        )
        
        # Health loss
        text_health_proj = self.text_health_proj(text_location_proj)
        image_health_proj = self.image_health_proj(image_location_proj)
        health_loss = self.supervised_loss(
            image_health_proj, text_health_proj, category, logit_scale
        )

        # Implicit category losses (only after first epoch)
        if epoch > 0:
            implicit_category0_loss = self.supervised_loss(
                image_features, text_features, implicit_category_0, logit_scale
            )
            implicit_category1_loss = self.supervised_loss(
                image_features, text_features, implicit_category_1, logit_scale
            )
        else:
            implicit_category0_loss = torch.tensor(0.0, device=device, requires_grad=True)
            implicit_category1_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            "contrastive_loss": contrastive_loss,
            "distill_loss": distill_loss,
            "location_loss": location_loss,
            "health_loss": health_loss,
            "implicit_category0_loss": implicit_category0_loss,
            "implicit_category1_loss": implicit_category1_loss
        }

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        teacher_image_features: torch.Tensor,
        teacher_text_features: torch.Tensor,
        location: torch.Tensor,
        category: torch.Tensor,
        implicit_category_0: torch.Tensor,
        implicit_category_1: torch.Tensor,
        epoch: int,
        logit_scale: torch.Tensor,
        output_dict: bool = False,
        image_names: Optional[List[str]] = None,
        health_names: Optional[List[str]] = None,
        location_names: Optional[List[str]] = None,
        idx_to_health_label: Optional[Dict[int, str]] = None,
        idx_to_location_label: Optional[Dict[int, str]] = None,
        save_results: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-task loss computation.
        
        Args:
            image_features: Student image features
            text_features: Student text features
            teacher_image_features: Teacher image features
            teacher_text_features: Teacher text features
            location: Location labels (one-hot)
            category: Health category labels (one-hot)
            implicit_category_0: First implicit category labels
            implicit_category_1: Second implicit category labels
            epoch: Current training epoch
            logit_scale: Logit scale parameter
            output_dict: Whether to return loss components as dictionary
            image_names: List of image names for logging
            health_names: List of health category names
            location_names: List of location names
            idx_to_health_label: Mapping from index to health label
            idx_to_location_label: Mapping from index to location label
            save_results: Whether to save classification results to CSV
            
        Returns:
            Total loss tensor or dictionary of loss components
        """
        # Compute all loss components
        loss_components = self.compute_loss_components(
            image_features, text_features, teacher_image_features, teacher_text_features,
            location, category, implicit_category_0, implicit_category_1, epoch, logit_scale
        )

        # Apply weights and compute total loss
        total_loss = sum(
            loss * self.loss_weights[name.replace('_loss', '')]
            for name, loss in loss_components.items()
        )

        # Save classification results if requested
        if (save_results and image_names is not None and 
            health_names is not None and location_names is not None and
            idx_to_health_label is not None and idx_to_location_label is not None):
            
            classification_results = self.get_classification_results(
                image_features, text_features, image_names, location, category, 
                implicit_category_0, implicit_category_1, logit_scale,
                health_names, location_names, idx_to_health_label, idx_to_location_label
            )
            
            self.save_results_to_csv(classification_results, epoch)

        # Return either total loss or loss components
        if output_dict:
            # Apply weights to individual components for reporting
            weighted_components = {
                name: loss * self.loss_weights[name.replace('_loss', '')]
                for name, loss in loss_components.items()
            }
            return weighted_components
        else:
            return total_loss

class HardNegativeLoss(nn.Module):
    """
    Hard Negative Noise Contrastive Estimation proposed in https://arxiv.org/abs/2301.02280
    beta1: hardness parameter for image features
    beta2: hardness parameter for text features
    alpha: the weighting function of the positive sample loss
    Setting alpha to 0, the loss is equivalent to the decoupled HN-NCE loss (DHN-NCE)
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, temperature=1.0,beta1=1.0, beta2 = 1.0, alpha=0.0, batch_size=1):
        super(HardNegativeLoss, self).__init__()
        self.temperature = temperature
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.batch_size = batch_size

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity between image and text features
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        mask = torch.eye(logits_per_image.size(0), dtype=torch.bool)
        mask = mask.to(image_features.device)

        # Positive pairs: diagonal elements
        pos = torch.exp(logits_per_image*mask)

        # Negative pairs: off-diagonal elements
        N = self.batch_size - 1

        neg_mask = ~mask

        # Calculate reweighting factors
        norm_term_img = torch.sum(torch.exp(logits_per_image*neg_mask),dim=-1)
        reweight_img = N * (torch.exp(self.beta1*logits_per_image*neg_mask))/norm_term_img
        norm_term_text = torch.sum(torch.exp(logits_per_text*neg_mask),dim=-1)
        reweight_text = N * (torch.exp(self.beta2*logits_per_text*neg_mask))/norm_term_text

        neg_img = reweight_img * torch.exp(logits_per_image*neg_mask)
        neg_text = reweight_text * torch.exp(logits_per_text*neg_mask)

        # Calculate loss
        loss = -torch.log(pos / (pos*self.alpha + neg_img)) -torch.log(pos / (pos*self.alpha + neg_text))

        return {"contrastive_loss": loss.mean()} if output_dict else loss.mean()


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


class SupervisedContrastiveLoss(ClipLoss):
    """
    Args:
        temperature: 温度参数，控制分布的平滑度（对应公式中的 T）
        base_temperature: 基准温度，用于稳定性保证
        scale_by_temperature: 是否通过温度缩放损失值
    """
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            temperature=0.07, 
            base_temperature=0.1, 
            scale_by_temperature=False
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.scale_by_temperature = scale_by_temperature
    
    def forward(self, image_features, text_features, labels, logit_scale, output_dict=False):
        device = image_features.device
        batch_size = image_features.shape[0]
        
        single_labels = torch.argmax(labels, dim=1)
        mask = (single_labels.unsqueeze(0) == single_labels.unsqueeze(1)).float()
        mask = mask - torch.eye(batch_size, device=device)
        mask = mask.clamp(min=0)

        pos_num = mask.sum(dim=1).clamp(min=1)  # 确保至少有一个正样本

        # 计算原始logits
        logits_per_image, logits_per_text = super().get_logits(image_features, text_features, logit_scale)

        # 温度缩放
        if self.scale_by_temperature:
            logits_per_image /= (self.temperature * self.base_temperature)
            logits_per_text /= (self.temperature * self.base_temperature)

        # ===== 关键修复：数值稳定计算 =====
        def stable_compute(logits, mask):
            # 1. 减最大值防止指数爆炸
            logits_max = logits.max(dim=1, keepdim=True).values.detach()
            stable_logits = logits - logits_max
            
            # 2. 计算稳定指数
            exp_logits = torch.exp(stable_logits)
            
            # 3. 安全计算分母（防止零除）
            diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
            exp_logits_no_diag = exp_logits.masked_fill(diag_mask, 0.0)
            
            epsilon = 1e-5 
            denominator = exp_logits_no_diag.sum(dim=1) + epsilon  # 添加小常数

            # 4. 计算分子
            numerator = (exp_logits * mask).sum(dim=1) + epsilon
            ratio = numerator / denominator
            ratio = torch.clamp(ratio, min=1e-8, max=1.0)
            # 5. 计算损失
            losses = -torch.log(numerator / denominator) / pos_num
            return losses.mean()

        # 计算双分支损失
        loss_image = stable_compute(logits_per_image, mask)
        loss_text = stable_compute(logits_per_text, mask.T)  # 注意文本分支用转置mask

        total_loss = (loss_image + loss_text) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss

#显式类别分类投影头
class Projection_head(nn.Module):
    def __init__(self, input_dim, output_dim=128, hidden_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        x = F.normalize(x, dim=-1, p=2)
        return x


class cls_CMCLIPLoss(ClipLoss):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features, cls_logits, label, output_dict=False):
        device = image_features.device
        # one-hot编码转成类别索引
        label = torch.argmax(label, dim=1)
        cls_loss = self.criterion(cls_logits,label)
        
        return {
            "cls_loss": cls_loss
            } if output_dict else cls_loss
