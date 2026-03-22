import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

from open_clip_train.visual import visual_eval
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def get_flatten_params(model, param_filter=None):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and (param_filter is None or param_filter(name)):
            params.append(param.detach().cpu().view(-1))
    if params:
        return torch.cat(params)
    else:
        return None
def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
def grad_norm(params):
    return (sum((p.grad.data.norm()**2 for p in params if p.grad is not None))**0.5)

def get_task_vector(model, pretrain_weights, param_filter=None):
    task_vec = []
    for name, param in model.named_parameters():
        if param.requires_grad and (param_filter is None or param_filter(name)):
            delta = (param.detach().cpu() - pretrain_weights[name])
            task_vec.append(delta.view(-1))
    if task_vec:
        return torch.cat(task_vec)
    else:
        return None
    

def project_gradients_to_orthogonal(model, vector, param_filter=None, eps=1e-8):
    """
    将模型参数的梯度投影到与给定向量正交的子空间。
    """
    if vector is None:
        return

    grad_vec = []
    param_shapes = []
    valid_params = []
    
    # 第一遍：收集所有需要处理的参数信息
    for name, param in model.named_parameters():
        if param_filter is None or param_filter(name):
            valid_params.append((name, param))
            if param.grad is not None:
                grad_vec.append(param.grad.view(-1))
                param_shapes.append(param.grad.shape)
            else:
                grad_vec.append(torch.zeros_like(param).view(-1))
                param_shapes.append(param.shape)
    
    if not grad_vec:
        return

    g = torch.cat(grad_vec)
    v = vector.detach()

    # 计算投影
    dot_product_gv = torch.dot(g, v)
    dot_product_vv = torch.dot(v, v) + eps
    scale = dot_product_gv / dot_product_vv
    g_proj = g - scale * v

    # 第二遍：将投影后的梯度重新分配回模型参数
    offset = 0
    for (name, param), shape in zip(valid_params, param_shapes):
        numel = param.numel()
        param.grad = g_proj[offset:offset + numel].view(shape)
        offset += numel
    
def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, pretrain_weights, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    last_teacher_update_step = -1
    
    if args.accum_freq > 1:
        accum_images, accum_texts, accum_img_name, accum_features = [], [], [], {}
        accum_category, accum_location, accum_health_name, accum_location_name = [], [], [], []
        accum_implicit_category0, accum_implicit_category1 = [], []
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, category, location, img_name, health_name, location_name = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        category = category.to(device=device, non_blocking=True)
        location = location.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        # 注意：这个被用来做下游分类训练
        if args.accum_freq == 1 or args.cls_model:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out.pop("logit_scale")
                if "logit_bias" in model_out:
                        logit_bias = model_out.pop("logit_bias")
                        
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, label=category, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss
                backward(total_loss, scaler)
                # # 调试信息：检查 cls_head 参数状态和梯度（在 backward 之后）
                # try:
                #     # 参数是否需要梯度
                #     requires = model.cls_head.weight.requires_grad
                #     # 检查 cls_head 参数是否在 optimizer 中
                #     in_optimizer = False
                #     for g in optimizer.param_groups:
                #         for p in g.get("params", []):
                #             if p is model.cls_head.weight:
                #                 in_optimizer = True
                #                 break
                #         if in_optimizer:
                #             break
                #     # 打印梯度信息
                #     if model.cls_head.weight.grad is None:
                #         logging.info(f"cls_head.weight.grad is None after backward; requires_grad={requires}, in_optimizer={in_optimizer}")
                #     else:
                #         logging.info(f"cls_head.weight.grad norm={model.cls_head.weight.grad.norm().item():.6f}; requires_grad={requires}, in_optimizer={in_optimizer}")
                # except Exception as e:
                #     logging.exception(f"Error while logging cls_head grad info: {e}")
        else:
            
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                accum_images.append(images)
                accum_texts.append(texts)
                accum_category.append(category)
                accum_location.append(location)
                accum_img_name.append(img_name)
                accum_health_name.append(health_name)
                accum_location_name.append(location_name)
                if ((i + 1) % args.accum_freq) == 0:
                    accum_category = torch.cat(accum_category)
                    accum_location = torch.cat(accum_location)
                    if epoch > -1:
                        model.update_implicit_components(args, epoch, accum_texts, accum_location)
                    with autocast():
                        for j in range(args.accum_freq):
                            img = accum_images[j]
                            txt = accum_texts[j]
                            model_out = model(img, txt)

                            for f in ("logit_scale", "logit_bias"):
                                model_out.pop(f, None)

                            for key, val in model_out.items():
                                if key in accum_features:
                                    accum_features[key].append(val)
                                else:
                                    accum_features[key] = [val]
                
            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            
            optimizer.zero_grad()
            # 每个epoch更新一次kmean和PCA
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]

                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])
                    inputs["location"] = accum_location
                    inputs["category"] = accum_category
                    
                    # accum_*_name 是按 batch 累积的列表（每项为一个 batch 的 name 列表）
                    # 需要展开为按样本的一维列表以匹配 image_features 的第 0 维
                    def _flatten_names(x):
                        if isinstance(x, list) and x and isinstance(x[0], (list, tuple)):
                            return [elem for batch in x for elem in batch]
                        return x

                    flat_img_names = _flatten_names(accum_img_name)
                    flat_health_names = _flatten_names(accum_health_name)
                    flat_location_names = _flatten_names(accum_location_name)

                    losses = loss(
                        **inputs,
                        **inputs_no_accum,
                        epoch=epoch,
                        output_dict=True,
                        image_names=flat_img_names,
                        health_names=flat_health_names,
                        location_names=flat_location_names,
                        idx_to_health_label=data["train"].dataloader.dataset.idx_to_health_label,
                        idx_to_location_label=data["train"].dataloader.dataset.idx_to_location_label,
                        save_results=True
                    )
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                    
                    explicit_loss = losses["location_loss"] + losses["health_loss"]
                    implicit_loss = losses["implicit_category0_loss"] + losses["implicit_category1_loss"]
                    distill_loss = losses["distill_loss"]
                    contrastive_loss = losses["contrastive_loss"]
                    optimizer.zero_grad()
                    # 保存显式任务梯度
                    scaler.scale(explicit_loss).backward(retain_graph=True)
                    explicit_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
                    optimizer.zero_grad()
                    # 保存隐式任务梯度
                    scaler.scale(implicit_loss).backward(retain_graph=True)
                    implicit_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
                    optimizer.zero_grad()
                    # 保存蒸馏任务梯度
                    scaler.scale(distill_loss).backward(retain_graph=True)
                    distill_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
                    optimizer.zero_grad()
                    # 保存对比学习任务梯度
                    scaler.scale(contrastive_loss).backward(retain_graph=True)
                    contrastive_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
                    optimizer.zero_grad()
                    
                    for p, g_exp, g_imp, g_distill, g_constr in zip(model.parameters(), explicit_grads, implicit_grads, distill_grads, contrastive_grads):
                        if p.requires_grad:
                            g_exp_flat = g_exp.flatten()
                            g_imp_flat = g_imp.flatten()
                            g_distill_flat = g_distill.flatten()
                            g_constr_flat = g_constr.flatten()
                            proj = torch.dot(g_imp_flat, g_exp_flat) / (g_exp_flat.norm() ** 2 + 1e-8)
                            g_orth = g_imp_flat - proj * g_exp_flat
                            # 合成最终梯度
                            # 计算当前批次的最终梯度
                            current_batch_grad = (g_exp_flat + g_orth + g_distill_flat + g_constr_flat).view_as(p)
                            
                            # 累加到全局梯度上 (这是梯度累积的关键！)
                            if p.grad is None:
                                p.grad = current_batch_grad
                            else:
                                p.grad += current_batch_grad

                if j == 0:
                    contrib_dict = {}
                    for k, v in losses.items():
                        if k == "loss" or v == torch.tensor(0.0):
                            continue
                        optimizer.zero_grad()
                        if scaler is not None:
                            scaler.scale(v).backward(retain_graph=True)
                        else:
                            v.backward(retain_graph=True)
                        contrib_dict[k] = grad_norm(model.text.parameters())
                        optimizer.zero_grad()
                    logging.info(
                        f"main_grad={contrib_dict.get('contrastive_loss', 0):.6f}, "
                        f"distill_grad={contrib_dict.get('distill_loss', 0):.6f}, "
                        f"location_grad={contrib_dict.get('location_loss', 0):.6f}, "
                        f"health_grad={contrib_dict.get('health_loss', 0):.6f}, "
                        f"implicit_category0_grad={contrib_dict.get('implicit_category0_loss', 0):.6f}, "
                        f"implicit_category1_grad={contrib_dict.get('implicit_category1_loss', 0):.6f}"
                    )
                # backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_img_name, accum_features = [], [], [], {}
            accum_category, accum_location, accum_health_name, accum_location_name = [], [], [], []
            accum_implicit_category0, accum_implicit_category1 = [], []
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        # teacher model update
        if args.force_CMCLIP and step > last_teacher_update_step and step % args.teacher_update_freq == 0:
            model.update_teacher()
            last_teacher_update_step = step
            logging.info(f"Updated teacher model at step {step}")
            
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, losses, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    # zero_shot class eval
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    # 仅在需要生成注意力可视化时，准备保存少量样本（防止复用/耗尽 val dataloader）
    vis_enabled = True
    vis_max_images = 64
    saved_vis_batches = []  # will store tuples of (images_cpu, texts_cpu, category_cpu, location_cpu)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts, category, location, *_ = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                category = category.to(device=device, non_blocking=True)
                location = location.to(device=device, non_blocking=True)
                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    teacher_image_features = model_out.get("teacher_image_features", None)
                    teacher_text_features = model_out.get("teacher_text_features", None)
                    implicit_category_0 = model_out.get("implicit_category_0", None)
                    implicit_category_1 = model_out.get("implicit_category_1", None)
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    loss = losses(image_features, text_features, teacher_image_features, teacher_text_features, location, category, implicit_category_0, implicit_category_1, epoch, logit_scale, output_dict=True)
                    contrastive_loss = loss["contrastive_loss"]

                    gen_loss = maybe_compute_generative_loss(model_out)

                # 保存少量用于可视化的样本（保存在 CPU，避免影响后续计算）
                if vis_enabled and len(saved_vis_batches) < vis_max_images:
                    try:
                        saved_vis_batches.append((
                            images.detach().cpu(),
                            texts.detach().cpu(),
                            category.detach().cpu(),
                            location.detach().cpu()
                        ))
                    except Exception:
                        # 若 detatch 失败则跳过该样本的保存
                        pass
                
                cumulative_loss += contrastive_loss * images.shape[0]
                num_samples += images.shape[0]
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * images.shape[0]
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # image and text retrieval eval
            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, **zero_shot_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    # ========== 在评估时生成4类可视化图像（Grad-CAM, SHAP Occlusion, Token-level, Text-guided CLS） ==========
    # 仅在主进程并且 args.generate_attention 为真时执行
    # if vis_enabled and 'val' in data:
    #     visual_eval( model, data, epoch, args, tokenizer, vis_max_images = 8)

    # =====================================================================

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 2, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

def cls_evaluate(model, loss_fn, data, epoch, args, tb_writer=None, tokenizer=None):
    """
    纯分类任务评估函数，专注于分类性能指标计算
    """
    metrics = {}
    # 仅主进程执行评估
    if not is_master(args):
        return metrics
    
    device = torch.device(args.device)
    model.eval()
    
    # 只处理验证集数据
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # 累计损失和预测结果存储
        total_loss = 0.0
        all_predictions = []
        all_true_labels = []
        
        with torch.inference_mode():  # 关闭梯度计算
            for batch_idx, batch in enumerate(dataloader):
                # 解析批次数据（图像、文本和标签）
                images, texts, category, location = batch[:4]
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                # 选择分类标签
                labels = category
                
                labels = labels.to(device=device, non_blocking=True)
                
                # 模型前向传播
                model_out = model(images, texts)
                logit_scale = model_out.pop("logit_scale")
                if "logit_bias" in model_out:
                    logit_bias = model_out.pop("logit_bias")
                batch_loss = loss_fn(**model_out, label=labels, output_dict=True)
                
                # 计算损失
                total_loss += sum(batch_loss.values()) * images.size(0)
                
                # 记录预测结果和真实标签
                predictions = torch.argmax(model_out["cls_logits"], dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                true_labels = torch.argmax(labels, dim=1)
                all_true_labels.extend(true_labels.cpu().numpy())
                
                # 更新样本计数
                num_samples += images.size(0)
                
                # 打印进度信息
                if (batch_idx % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples}/{samples_per_val}]\t"
                        f"Current Loss: {total_loss/num_samples:.6f}"
                    )

        # 计算基础指标
        avg_loss = total_loss / num_samples
        metrics["val_loss"] = avg_loss
        metrics["val_accuracy"] = accuracy_score(all_true_labels, all_predictions)
        
        # 计算每类详细指标
        class_report = classification_report(
            all_true_labels, 
            all_predictions, 
            output_dict=True, 
            zero_division=0
        )
        
        # 提取每类的精确率、召回率和F1分数
        for cls_name, stats in class_report.items():
            if cls_name not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics[f"val_precision_{cls_name}"] = stats['precision']
                metrics[f"val_recall_{cls_name}"] = stats['recall']
                metrics[f"val_f1_{cls_name}"] = stats['f1-score']
        
        # 计算混淆矩阵对角线元素（每类正确分类数）
        cm = confusion_matrix(all_true_labels, all_predictions)
        for i in range(cm.shape[0]):
            metrics[f"val_correct_{i}"] = cm[i, i]
        
        # 记录评估信息
        metrics["epoch"] = epoch
        metrics["total_samples"] = num_samples

    # 无评估数据时直接返回
    if not metrics:
        return metrics

    # 打印评估结果摘要
    logging.info(
        f"Eval Summary - Epoch {epoch}\t"
        f"Loss: {metrics['val_loss']:.4f}\t"
        f"Accuracy: {metrics['val_accuracy']:.4f}"
    )

    # 日志记录与可视化
    if args.save_logs:
        # 先生成一个可序列化的 metrics 副本，避免 torch.Tensor / numpy 类型导致 json.dumps 失败
        serializable_metrics = {}
        for k, v in metrics.items():
            # torch.Tensor -> 原生标量或 list
            if isinstance(v, torch.Tensor):
                try:
                    serializable_metrics[k] = v.item()
                except Exception:
                    serializable_metrics[k] = v.detach().cpu().tolist()
            # numpy 标量 -> 原生标量
            elif isinstance(v, np.generic):
                serializable_metrics[k] = v.item()
            # numpy 数组 -> 列表
            elif isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            else:
                # 已经是原生类型（float/int/str 等），直接使用
                serializable_metrics[k] = v

        # TensorBoard日志（使用可序列化版本）
        if tb_writer is not None:
            for name, value in serializable_metrics.items():
                if name not in ['epoch', 'total_samples']:
                    # 确保传入的是标量
                    if isinstance(value, (list, dict)):
                        # 跳过复杂结构
                        continue
                    tb_writer.add_scalar(f"val/{name}", value, epoch)

        # 保存到JSONL文件（使用可序列化版本）
        with open(os.path.join(args.checkpoint_path, "classification_results.jsonl"), "a+") as f:
            f.write(json.dumps(serializable_metrics) + "\n")

    # WandB可视化
    if args.wandb:
        assert wandb is not None, 'Please install wandb to use this feature.'
        # 计算训练步数（如果需要）
        step = None
        if 'train' in data:
            train_loader = data['train'].dataloader
            num_batches = train_loader.num_batches // args.accum_freq
            step = num_batches * epoch
        
        # 记录wandb日志（使用可序列化版本）
        wandb_log = {f"val/{k}": v for k, v in serializable_metrics.items()}
        wandb.log(wandb_log, step=step)

    return metrics


