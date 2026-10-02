import json
import logging
import math
import os
import time

import numpy as np
import torch
try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from open_clip_train import grad_methods


class AverageMeter(object):
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


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    implicit_updated_this_epoch = False

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}
        accum_category, accum_location = [], []
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, category, location = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        category = category.to(device=device, non_blocking=True)
        location = location.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out.pop("logit_scale")
                if "logit_bias" in model_out:
                    logit_bias = model_out.pop("logit_bias")

                if args.CMCLIP_loss or args.force_CMCLIP:
                    losses = loss(**model_out, location=location, category=category, output_dict=True)
                else:
                    losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss
                backward(total_loss, scaler)
        else:
            with torch.no_grad():
                accum_images.append(images)
                accum_texts.append(texts)
                accum_category.append(category)
                accum_location.append(location)
                if ((i + 1) % args.accum_freq) == 0:
                    accum_category = torch.cat(accum_category)
                    accum_location = torch.cat(accum_location)
                    if epoch > args.implicit_start_epoch and not implicit_updated_this_epoch \
                            and args.imp_weight > 0:
                        model.update_implicit_supervision(args, epoch, accum_texts, accum_location)
                        implicit_updated_this_epoch = True
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    with autocast():
                        for j in range(args.accum_freq):
                            img = accum_images[j]
                            txt = accum_texts[j]
                            model_out = model(img, txt, epoch)

                            for f in ("logit_scale", "logit_bias"):
                                model_out.pop(f, None)

                            for key, val in model_out.items():
                                if key in accum_features:
                                    accum_features[key].append(val)
                                else:
                                    accum_features[key] = [val]

            if ((i + 1) % args.accum_freq) > 0:
                continue

            optimizer.zero_grad()
            params = list(model.parameters())
            param_shapes = [p.shape for p in params]
            task_grads_accum = {"explicit": None, "implicit": None, "contrastive": None}
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]

                with autocast():
                    model_out = model(images, texts, epoch)

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

                    losses = loss(
                        **inputs,
                        **inputs_no_accum,
                        epoch=epoch,
                        output_dict=True,
                        save_results=True
                    )
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                    explicit_loss = losses["location_loss"] + losses["health_loss"]
                    implicit_loss = losses["implicit_category0_loss"] + losses["implicit_category1_loss"]
                    contrastive_loss = losses["contrastive_loss"]

                    task_losses = {
                        "explicit": explicit_loss,
                        "implicit": implicit_loss,
                        "contrastive": contrastive_loss,
                    }

                    for name, tl in task_losses.items():
                        optimizer.zero_grad()
                        if scaler is not None:
                            scaler.scale(tl).backward(retain_graph=True)
                        else:
                            tl.backward(retain_graph=True)
                        flat = grad_methods.flatten(
                            [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                        )
                        task_grads_accum[name] = flat if task_grads_accum[name] is None else task_grads_accum[name] + flat
                    optimizer.zero_grad()

            combined_flat = grad_methods.combine(
                task_grads_accum, args.grad_method,
            )
            for p, g in zip(params, grad_methods.unflatten(combined_flat, param_shapes)):
                if p.requires_grad:
                    p.grad = g

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

        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}
            accum_category, accum_location = [], []
        with torch.no_grad():
            _m = unwrap_model(model)
            _m.logit_scale.clamp_(0, math.log(getattr(_m, 'logit_scale_max', 100.0)))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            loss_log = " ".join(f"{key.replace('_loss', '')} {m.avg:.4f}" for key, m in losses_m.items())
            logging.info(
                f"train epoch {epoch} [{percent_complete:.0f}%] "
                f"lr {optimizer.param_groups[0]['lr']:.2e} "
                f"scale {logit_scale.item():.3f} | {loss_log}"
            )

            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.accum_freq * args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.avg for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step
                wandb.log(log_data, step=step)

            batch_time_m.reset()
            data_time_m.reset()


def evaluate(model, losses, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0

        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts, category, location, *_ = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                category = category.to(device=device, non_blocking=True)
                location = location.to(device=device, non_blocking=True)
                with autocast():
                    model_out = model(images, texts, epoch)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    implicit_category_0 = model_out.get("implicit_category_0", None)
                    implicit_category_1 = model_out.get("implicit_category_1", None)
                    logit_scale = model_out["logit_scale"]
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    loss = losses(image_features, text_features,
                                  image_location_proj=model_out.get("image_location_proj"),
                                  text_location_proj=model_out.get("text_location_proj"),
                                  image_health_proj=model_out.get("image_health_proj"),
                                  text_health_proj=model_out.get("text_health_proj"),
                                  location=location, category=category,
                                  implicit_category_0=implicit_category_0,
                                  implicit_category_1=implicit_category_1,
                                  epoch=epoch, logit_scale=logit_scale,
                                  output_dict=True)
                    contrastive_loss = loss["contrastive_loss"]

                cumulative_loss += contrastive_loss * images.shape[0]
                num_samples += images.shape[0]
            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            val_loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, **zero_shot_metrics, "clip_val_loss": val_loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"eval epoch {epoch} | "
        + " ".join(f"{k} {round(v, 4):.4f}" for k, v in metrics.items() if k not in ('epoch', 'num_samples'))
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


