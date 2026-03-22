import os
import logging
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from open_clip_train.precision import get_autocast
from open_clip import get_input_dtype
def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def visual_eval(model, data, epoch, args, tokenizer=None, vis_max_images=16):
    saved_vis_batches = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    if 'val' not in data:
        return {}
    dataloader = data['val'].dataloader
    
    texts_list = []
    for i, batch in enumerate(dataloader):
        images, texts, category, location = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        category = category.to(device=device, non_blocking=True)
        location = location.to(device=device, non_blocking=True)
        if len(saved_vis_batches) < vis_max_images:
            saved_vis_batches.append((
                images, 
                texts, 
                category, 
                location
            ))
            texts_list.append(texts)
        else:
            break
    toks = torch.cat(texts_list, dim=0)
    
    with torch.inference_mode():
        with autocast():
            model_unwrapped = unwrap_model(model)
            if hasattr(model_unwrapped, "encode_text"):
                text_features = model_unwrapped.encode_text(toks)
    # grad_cam
    # grad_cam_path = visualize_gradcam(model,saved_vis_batches, args, epoch, out_dir=None, max_images=max_images)

    # shap
    # shap_path = visualize_shap(model, val_vis_loader, args, epoch, out_dir=None, max_images=max_images)
    
    # token level
    # token_paths = visualize_token_level(model, val_vis_loader, args, epoch, out_dir=None, max_images=max_images)

    # text guided cls
    cls_paths = visualize_text_guided_cls(
                    model,
                    saved_vis_batches,
                    args,
                    epoch,
                    class_texts=texts_list,
                    class_text_features=text_features,
                    tokenizer=tokenizer,
                    out_dir=None,
                    max_images=vis_max_images,
                )
    
    logging.info(
        f"Saved attention visualizations: cls={len(cls_paths)}"
    )
    # =====================================================================

def _find_conv_layer(mod):
    for _, m in reversed(list(mod.named_modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None

def _restore_image_tensor(img_t, model, visual):
    """从标准化张量恢复为PIL图像（尝试从 model/visual 获取 mean/std，否则用 ImageNet）"""
    mean = getattr(model, "image_mean", None) or getattr(visual, "image_mean", None)
    std = getattr(model, "image_std", None) or getattr(visual, "image_std", None)
    if mean is None or std is None:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = torch.clamp(img_t * std + mean, 0, 1)
    return TF.to_pil_image(img)

def visualize_gradcam(model, dataloader, args, epoch, out_dir=None, max_images=16):
    """
    对 Conv backbone 使用 Grad-CAM（最后一个 Conv2d 层）。
    返回保存路径列表。
    """
    device = torch.device(args.device)
    model = unwrap_model(model)
    visual = getattr(model, "visual", None)
    if visual is None:
        logging.warning("visual not found for gradcam")
        return []
    conv = _find_conv_layer(visual)
    if conv is None:
        logging.warning("no conv layer found for gradcam")
        return []

    if out_dir is None:
        out_dir = os.path.join(args.checkpoint_path, f"gradcam_epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)

    cache = {"acts": None, "grads": None}
    def fh(module, inp, out): cache["acts"] = out.detach()
    def bh(module, grad_in, grad_out):
        g = grad_out[0] if isinstance(grad_out, (list, tuple)) else grad_out
        if g is not None: cache["grads"] = g.detach()

    fh_h = conv.register_forward_hook(fh)
    try:
        bh_h = conv.register_full_backward_hook(bh)
    except Exception:
        bh_h = conv.register_backward_hook(bh)

    saved = 0
    saved_paths = []
    try:
        model.eval()
        with torch.enable_grad():
            for batch in dataloader:
                images, texts, *_ = batch[:4]
                images = images.to(device=device)
                texts = texts.to(device=device)
                out = model(images, texts)
                if isinstance(out, dict):
                    img_feats = out.get("image_features", None)
                    txt_feats = out.get("text_features", None)
                    logit_scale = out.get("logit_scale", torch.tensor(1.0, device=device))
                else:
                    img_feats = out[0] if len(out) > 0 else None
                    txt_feats = out[1] if len(out) > 1 else None
                    logit_scale = out[2] if len(out) > 2 else torch.tensor(1.0, device=device)
                if img_feats is None or txt_feats is None:
                    continue
                sims = (img_feats * txt_feats).sum(dim=1) * logit_scale.view(-1)
                bs = images.size(0)
                for i in range(bs):
                    if saved >= max_images:
                        break
                    cache["acts"] = None; cache["grads"] = None
                    model.zero_grad()
                    sims[i].backward(retain_graph=True)
                    acts = cache["acts"]; grads = cache["grads"]
                    if acts is None or grads is None:
                        continue
                    act = acts[i]
                    grad = grads[i]
                    if act.ndim == 4:
                        act = act.squeeze(0); grad = grad.squeeze(0)
                    weights = grad.mean(dim=(1,2), keepdim=True)
                    cam = (weights * act).sum(dim=0).relu()
                    cam = cam - cam.min()
                    if cam.max() > 0:
                        cam = cam / cam.max()
                    cam_np = cam.cpu().numpy()
                    pil = _restore_image_tensor(images[i].detach().cpu(), model, visual)
                    heat = Image.fromarray((cam_np * 255).astype("uint8")).resize(pil.size, resample=Image.BILINEAR).convert("L")
                    cmap = plt.get_cmap("jet")
                    heat_colored = Image.fromarray((cmap(np.array(heat)/255.0)[...,:3]*255).astype("uint8"))
                    overlay = Image.blend(pil.convert("RGBA"), heat_colored.convert("RGBA"), alpha=0.5)
                    path = os.path.join(out_dir, f"gradcam_epoch{epoch}_img{saved}.png")
                    overlay.convert("RGB").save(path)
                    saved_paths.append(path)
                    saved += 1
                if saved >= max_images:
                    break
    finally:
        try: fh_h.remove()
        except Exception: pass
        try: bh_h.remove()
        except Exception: pass

    logging.info(f"Saved {len(saved_paths)} gradcam images to {out_dir}")
    return saved_paths

def visualize_shap(model, dataloader, args, epoch, out_dir=None, max_images=8, occlusion_size=32, baseline=0.0):
    """
    使用基于遮挡的近似 Shapley (occlusion) 生成每张图的贡献热力图。
    - occlusion_size: 遮挡窗口大小（像素）
    - baseline: 遮挡值 (0->黑, 或均值)
    返回保存路径列表。
    """
    device = torch.device(args.device)
    model = unwrap_model(model)
    visual = getattr(model, "visual", None)
    if visual is None:
        logging.warning("visual not found for shap")
        return []

    if out_dir is None:
        out_dir = os.path.join(args.checkpoint_path, f"shap_epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    saved_paths = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images, texts, *_ = batch[:4]
            images = images.to(device=device)
            texts = texts.to(device=device)
            # baseline score (original)
            out = model(images, texts)
            if isinstance(out, dict):
                img_feats = out.get("image_features", None)
                txt_feats = out.get("text_features", None)
                logit_scale = out.get("logit_scale", torch.tensor(1.0, device=device))
            else:
                img_feats = out[0] if len(out) > 0 else None
                txt_feats = out[1] if len(out) > 1 else None
                logit_scale = out[2] if len(out) > 2 else torch.tensor(1.0, device=device)
            if img_feats is None or txt_feats is None:
                continue
            bs, _, H, W = images.shape
            orig_sims = ((img_feats * txt_feats).sum(dim=1) * logit_scale.view(-1)).detach().cpu().numpy()
            for b in range(bs):
                if saved >= max_images:
                    break
                img = images[b:b+1].clone()
                base_score = orig_sims[b]
                # build contribution map
                n_h = max(1, math.ceil(H / occlusion_size))
                n_w = max(1, math.ceil(W / occlusion_size))
                contrib = np.zeros((n_h, n_w), dtype=np.float32)
                for iy in range(n_h):
                    for ix in range(n_w):
                        y0 = int(iy * occlusion_size)
                        x0 = int(ix * occlusion_size)
                        y1 = min(H, y0 + occlusion_size)
                        x1 = min(W, x0 + occlusion_size)
                        img_masked = img.clone()
                        img_masked[..., :, y0:y1, x0:x1] = baseline
                        out_m = model(img_masked.to(device=device), texts[b:b+1])
                        if isinstance(out_m, dict):
                            imgf_m = out_m.get("image_features", None)
                            txtf = out_m.get("text_features", None)
                            logit_m = out_m.get("logit_scale", torch.tensor(1.0, device=device))
                        else:
                            imgf_m = out_m[0] if len(out_m) > 0 else None
                            txtf = out_m[1] if len(out_m) > 1 else None
                            logit_m = out_m[2] if len(out_m) > 2 else torch.tensor(1.0, device=device)
                        if imgf_m is None or txtf is None:
                            score_m = 0.0
                        else:
                            score_m = ((imgf_m * txtf).sum(dim=1) * logit_m.view(-1)).detach().cpu().numpy()[0]
                        contrib[iy, ix] = base_score - score_m
                # normalize and upsample
                cmap_map = contrib
                cmap_map = cmap_map - cmap_map.min()
                if cmap_map.max() > 0: cmap_map = cmap_map / cmap_map.max()
                # upsample to image size
                pil = _restore_image_tensor(images[b].detach().cpu(), model, visual)
                heat = Image.fromarray((np.kron(cmap_map, np.ones((occlusion_size//1, occlusion_size//1))) * 255).astype("uint8"))
                heat = heat.resize(pil.size, resample=Image.BILINEAR).convert("L")
                cmap = plt.get_cmap("jet")
                heat_colored = Image.fromarray((cmap(np.array(heat)/255.0)[...,:3]*255).astype("uint8"))
                overlay = Image.blend(pil.convert("RGBA"), heat_colored.convert("RGBA"), alpha=0.5)
                path = os.path.join(out_dir, f"shap_epoch{epoch}_img{saved}.png")
                overlay.convert("RGB").save(path)
                saved_paths.append(path)
                saved += 1
            if saved >= max_images:
                break
    logging.info(f"Saved {len(saved_paths)} shap (occlusion) images to {out_dir}")
    return saved_paths

def visualize_token_level(model, dataloader, args, epoch, out_dir=None, max_images=16):
    """
    对 ViT 使用 token-level 通道平均生成热力图（不反向）。
    """
    device = torch.device(args.device)
    model = unwrap_model(model)
    visual = getattr(model, "visual", None)
    if visual is None or not (hasattr(visual, "conv1") and hasattr(visual, "positional_embedding")):
        logging.warning("visual not ViT-like for token-level")
        return []
    if out_dir is None:
        out_dir = os.path.join(args.checkpoint_path, f"token_epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)
    visual.output_tokens = True
    saved = 0
    saved_paths = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images, texts, *_ = batch[:4]
            images = images.to(device=device)
            outs = visual(images)
            if not (isinstance(outs, tuple) and len(outs) >= 2):
                logging.warning("Visual module did not return tokens. Skipping text-guided ViT visualization.")
                break
            
            # For timm ViT with output_tokens=True, outs is a tuple:
            # outs[0] is the CLS token embedding (B, C)
            # outs[1] is the patch tokens (B, P, C)
            tokens = outs[1] # (B, P, C)
            tokens = tokens / (tokens.norm(dim=-1, keepdim=True)+1e-8)
            sims = torch.einsum("bpc,kc->bpk", tokens, class_text_features.to(tokens.dtype).to(device))
            bs = images.size(0)
            for i in range(bs):
                if saved >= max_images:
                    break
                h = heat[i].detach().cpu().numpy()
                if side is not None:
                    h = h.reshape(side, side)
                else:
                    h = h.reshape(1, -1)
                h = (h - h.min()) / (h.max() - h.min() + 1e-8)
                pil = _restore_image_tensor(images[i].detach().cpu(), model, visual)
                heat_im = Image.fromarray((h * 255).astype("uint8")).resize(pil.size, resample=Image.BILINEAR).convert("L")
                cmap = plt.get_cmap("jet")
                heat_colored = Image.fromarray((cmap(np.array(heat_im)/255.0)[...,:3]*255).astype("uint8"))
                overlay = Image.blend(pil.convert("RGBA"), heat_colored.convert("RGBA"), alpha=0.5)
                path = os.path.join(out_dir, f"token_epoch{epoch}_img{saved}.png")
                overlay.convert("RGB").save(path)
                saved_paths.append(path)
                saved += 1
            if saved >= max_images:
                break
    visual.output_tokens = False
    logging.info(f"Saved {len(saved_paths)} token-level images to {out_dir}")
    return saved_paths

def visualize_text_guided_cls(model, dataloader, args, epoch, class_texts=None, class_text_features=None, tokenizer=None, out_dir=None, max_images=16):
    """
    根据类别文本生成热力图：
     - ViT: 用 patch-token 与 class 特征相似度（无需反向）
     - Conv: 使用 Grad-CAM per-class（对每个 class 以相似度为目标反向）
    返回保存路径列表。
    """
    device = torch.device(args.device)
    model = unwrap_model(model)
    visual = getattr(model, "visual", None)
    if visual is None:
        logging.warning("visual not found for text-guided")
        return []

    class_text_features = class_text_features.detach()
    class_text_features = class_text_features / (class_text_features.norm(dim=-1, keepdim=True) + 1e-8)

    is_vit = hasattr(visual, "conv1") and hasattr(visual, "positional_embedding")
    conv = _find_conv_layer(visual)

    if out_dir is None:
        out_dir = os.path.join(args.checkpoint_path, f"cls_attention_epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    saved_paths = []

    if is_vit:
        visual.output_tokens = True
        model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                images, texts, *_ = batch[:4]
                images = images.to(device=device)
                outs = visual(images)
                if not (isinstance(outs, tuple) and len(outs) >= 2):
                    logging.warning("Visual module did not return tokens. Skipping text-guided ViT visualization.")
                    break
                
                # For timm ViT with output_tokens=True, outs is a tuple:
                # outs[0] is the CLS token embedding (B, C)
                # outs[1] is the patch tokens (B, P, C)
                tokens = outs[1] # (B, P, C)
                tokens = tokens / (tokens.norm(dim=-1, keepdim=True)+1e-8)
                sims = torch.einsum("bpc,kc->bpk", tokens, class_text_features.to(tokens.dtype).to(device))
                bs = images.size(0)
                for i in range(bs):
                    if saved >= max_images:
                        break
                    pil = _restore_image_tensor(images[i].detach().cpu(), model, visual)
                    P = sims.shape[1]
                    side = int(P**0.5) if int(P**0.5)**2 == P else None
                    for k in range(class_text_features.shape[0]):
                        score_map = sims[i, :, k].detach().cpu().numpy()
                        if side is not None:
                            score_map = score_map.reshape(side, side)
                        else:
                            score_map = score_map.reshape(1, -1)
                        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
                        heat = Image.fromarray((score_map*255).astype("uint8")).resize(pil.size, resample=Image.BILINEAR).convert("L")
                        cmap = plt.get_cmap("jet")
                        heat_colored = Image.fromarray((cmap(np.array(heat)/255.0)[...,:3]*255).astype("uint8"))
                        overlay = Image.blend(pil.convert("RGBA"), heat_colored.convert("RGBA"), alpha=0.5)
                        path = os.path.join(out_dir, f"cls_epoch{epoch}_img{saved}_class{k}.png")
                        overlay.convert("RGB").save(path)
                        saved_paths.append(path)
                    saved += 1
                if saved >= max_images:
                    break
        visual.output_tokens = False
        logging.info(f"Saved {len(saved_paths)} text-guided ViT images to {out_dir}")
        return saved_paths

    if conv is None:
        logging.warning("no conv found for text-guided conv path")
        return []

    # conv path: gradcam per class
    cache = {"acts": None, "grads": None}
    def fh(module, inp, out): cache["acts"] = out.detach()
    def bh(module, grad_in, grad_out):
        g = grad_out[0] if isinstance(grad_out, (list,tuple)) else grad_out
        if g is not None: cache["grads"] = g.detach()
    fh_h = conv.register_forward_hook(fh)
    try:
        bh_h = conv.register_full_backward_hook(bh)
    except Exception:
        bh_h = conv.register_backward_hook(bh)

    try:
        model.eval()
        with torch.enable_grad():
            for batch in dataloader:
                images, texts, *_ = batch[:4]
                images = images.to(device=device)
                out = model(images, None) if hasattr(model, "encode_text") else model(images, texts)
                if isinstance(out, dict):
                    img_feats = out.get("image_features", None)
                else:
                    img_feats = out[0] if len(out) > 0 else None
                if img_feats is None:
                    continue
                img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True)+1e-8)
                bs = images.size(0)
                for i in range(bs):
                    if saved >= max_images:
                        break
                    pil = _restore_image_tensor(images[i].detach().cpu(), model, visual)
                    for k in range(class_text_features.shape[0]):
                        txt = class_text_features[k].to(img_feats.device)
                        score = (img_feats[i:i+1] * txt.view(1, -1)).sum()
                        cache["acts"] = None; cache["grads"] = None
                        model.zero_grad()
                        score.backward(retain_graph=True)
                        acts = cache["acts"]; grads = cache["grads"]
                        if acts is None or grads is None:
                            continue
                        act = acts[i]; grad = grads[i]
                        weights = grad.mean(dim=(1,2), keepdim=True)
                        cam = (weights * act).sum(dim=0).relu()
                        cam = cam - cam.min()
                        if cam.max()>0: cam = cam / cam.max()
                        cam_np = cam.cpu().numpy()
                        heat = Image.fromarray((cam_np*255).astype("uint8")).resize(pil.size, resample=Image.BILINEAR).convert("L")
                        cmap = plt.get_cmap("jet")
                        heat_colored = Image.fromarray((cmap(np.array(heat)/255.0)[...,:3]*255).astype("uint8"))
                        overlay = Image.blend(pil.convert("RGBA"), heat_colored.convert("RGBA"), alpha=0.5)
                        path = os.path.join(out_dir, f"cls_epoch{epoch}_img{saved}_class{k}.png")
                        overlay.convert("RGB").save(path)
                        saved_paths.append(path)
                    saved += 1
                if saved >= max_images:
                    break
    finally:
        try: fh_h.remove()
        except Exception: pass
        try: bh_h.remove()
        except Exception: pass

    logging.info(f"Saved {len(saved_paths)} text-guided conv images to {out_dir}")
    return saved_paths
