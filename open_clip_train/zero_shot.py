import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip_train.precision import get_autocast
import json
import numpy



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5
def med_run(model, classifier, cat_classifier, loc_classifier, dataloader, combo_to_idx, cat_to_idx, loc_to_idx, args):
    """
    评估函数，同时评估联合分类和独立分类性能
    
    参数:
        model: 基础模型
        classifier: 联合分类器 (category+location)
        cat_classifier: 类别单独分类器
        loc_classifier: 位置单独分类器
        dataloader: 数据加载器
        combo_to_idx: 联合类别到索引的映射
        cat_to_idx: 类别到索引的映射
        loc_to_idx: 位置到索引的映射
        args: 其他参数
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    # 创建反向映射，用于将独热编码转换回类别名称
    idx_to_cat = {v: k for k, v in cat_to_idx.items()}
    idx_to_loc = {v: k for k, v in loc_to_idx.items()}
    
    with torch.inference_mode():
        # 联合分类的准确率跟踪
        combo_top1, combo_top2, combo_top5, combo_top10, combo_n = 0., 0., 0., 0., 0.
        
        # 类别单独分类的准确率跟踪
        cat_top1, cat_top2, cat_top5, cat_top10, cat_n = 0., 0., 0., 0., 0.
        
        # 位置单独分类的准确率跟踪
        loc_top1, loc_top2, loc_top5, loc_top10, loc_n = 0., 0., 0., 0., 0.
        
        for images, _, category_onehot, location_onehot in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            batch_size = images.shape[0]
            
            # 将独热编码转换为类别名称
            category = []
            location = []
            for onehot in category_onehot:
                # 找到独热编码中值为1的索引
                cat_idx = torch.argmax(onehot).item()
                # 转换为类别名称
                cat_name = idx_to_cat.get(cat_idx, None)
                category.append(cat_name)
            for onehot in location_onehot:
                # 找到独热编码中值为1的索引
                loc_idx = torch.argmax(onehot).item()
                # 转换为位置名称
                loc_name = idx_to_loc.get(loc_idx, None)
                location.append(loc_name)
                
            # 准备联合分类的目标和掩码
            batch_combinations = list(zip(category, location))
            combo_target = []
            valid_mask = []
            for comb in batch_combinations:
                if comb in combo_to_idx:
                    combo_target.append(combo_to_idx[comb])
                    valid_mask.append(True)
                else:
                    print(f"Warning: 组合 {comb} 不在类别组合映射中，跳过该样本。")
                    valid_mask.append(False)  # 标记无效样本
            
            # 准备类别单独分类的目标
            cat_target = []
            for cat in category:
                if cat in cat_to_idx:
                    cat_target.append(cat_to_idx[cat])
                else:
                    print(f"Warning: 类别 {cat} 不在类别映射中")
                    cat_target.append(-1)  # 无效标记
            
            # 准备位置单独分类的目标
            loc_target = []
            for loc in location:
                if loc in loc_to_idx:
                    loc_target.append(loc_to_idx[loc])
                else:
                    print(f"Warning: 位置 {loc} 不在位置映射中")
                    loc_target.append(-1)  # 无效标记
            
            # 转换为张量并过滤无效样本
            combo_target = torch.tensor(combo_target, device=device)
            cat_target = torch.tensor(cat_target, device=device)
            loc_target = torch.tensor(loc_target, device=device)
            
            valid_images = images[valid_mask]
            valid_cat_target = cat_target[valid_mask]
            valid_loc_target = loc_target[valid_mask]
            valid_combo_target = combo_target[valid_mask]
            
            valid_batch_size = valid_images.shape[0]
            if valid_batch_size == 0:
                continue  # 没有有效样本，跳过当前批次
            
            with autocast():
                # 获取图像特征
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                valid_features = image_features[valid_mask]
                
                # 联合分类预测
                combo_logits = 100. * valid_features @ classifier
                
                # 类别单独分类预测
                cat_logits = 100. * valid_features @ cat_classifier
                
                # 位置单独分类预测
                loc_logits = 100. * valid_features @ loc_classifier
            
            # 计算联合分类准确率
            c_acc1, c_acc2, c_acc5, c_acc10 = accuracy(combo_logits, valid_combo_target, topk=(1, 2, 5, 10))
            combo_top1 += c_acc1
            combo_top2 += c_acc2
            combo_top5 += c_acc5
            combo_top10 += c_acc10
            combo_n += valid_batch_size
            
            # 计算类别单独分类准确率
            cat_acc1, cat_acc2, cat_acc5, cat_acc10 = accuracy(cat_logits, valid_cat_target, topk=(1, 2, 5, 10))
            cat_top1 += cat_acc1
            cat_top2 += cat_acc2
            cat_top5 += cat_acc5
            cat_top10 += cat_acc10
            cat_n += valid_batch_size
            
            # 计算位置单独分类准确率
            loc_acc1, loc_acc2, loc_acc5, loc_acc10 = accuracy(loc_logits, valid_loc_target, topk=(1, 2, 5, 8))
            loc_top1 += loc_acc1
            loc_top2 += loc_acc2
            loc_top5 += loc_acc5
            loc_top10 += loc_acc10
            loc_n += valid_batch_size

    # 计算平均准确率
    combo_results = (
        combo_top1 / combo_n if combo_n > 0 else 0,
        combo_top2 / combo_n if combo_n > 0 else 0,
        combo_top5 / combo_n if combo_n > 0 else 0,
        combo_top10 / combo_n if combo_n > 0 else 0
    )
    
    cat_results = (
        cat_top1 / cat_n if cat_n > 0 else 0,
        cat_top2 / cat_n if cat_n > 0 else 0,
        cat_top5 / cat_n if cat_n > 0 else 0,
        cat_top10 / cat_n if cat_n > 0 else 0
    )
    
    loc_results = (
        loc_top1 / loc_n if loc_n > 0 else 0,
        loc_top2 / loc_n if loc_n > 0 else 0,
        loc_top5 / loc_n if loc_n > 0 else 0,
        loc_top10 / loc_n if loc_n > 0 else 0
    )
    
    return combo_results, cat_results, loc_results


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'val' not in data and 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    with open(args.config, 'r', encoding='utf-8') as f:
        config=json.load(f)
    idx_to_classnames_0 = config['category']
    idx_to_classnames_1 = config['location']
    classnames_0 = [v for _, v in idx_to_classnames_0.items()]
    classnames_1 = [v for _, v in idx_to_classnames_1.items()]
    prompt_templates = config['student_prompt_template']
    disease_templates = config['disease_prompt_tempalte']
    location_templates = config['location_prompt_template']
    with autocast():
        joint_clf, combo_map, cat_clf, cat_map, loc_clf, loc_map = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            # classnames=IMAGENET_CLASSNAMES,
            # templates=OPENAI_IMAGENET_TEMPLATES,
            classnames_0=classnames_0,
            classnames_1=classnames_1,
            templates_joint=prompt_templates,
            templates_0=disease_templates,
            templates_1=location_templates,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'val' in data:
        combo_results, cat_results, loc_results = med_run(
            model, 
            joint_clf, 
            cat_clf, 
            loc_clf, 
            data['val'].dataloader, 
            combo_map, 
            cat_map, 
            loc_map, 
            args
        )
         # 保存联合分类结果
        results['med-zeroshot-val-joint-top1'] = combo_results[0]
        results['med-zeroshot-val-joint-top2'] = combo_results[1]
        results['med-zeroshot-val-joint-top5'] = combo_results[2]
        results['med-zeroshot-val-joint-top10'] = combo_results[3]
        
        # 保存类别单独分类结果
        results['med-zeroshot-val-category-top1'] = cat_results[0]
        results['med-zeroshot-val-category-top2'] = cat_results[1]
        results['med-zeroshot-val-category-top5'] = cat_results[2]
        results['med-zeroshot-val-category-top10'] = cat_results[3]
        
        # 保存位置单独分类结果
        results['med-zeroshot-val-location-top1'] = loc_results[0]
        results['med-zeroshot-val-location-top2'] = loc_results[1]
        results['med-zeroshot-val-location-top5'] = loc_results[2]
        results['med-zeroshot-val-location-top10'] = loc_results[3]
        
        # 打印结果摘要
        logging.info(f"Validation Results - Joint: Top1={combo_results[0]:.4f}, Top5={combo_results[2]:.4f}")
        logging.info(f"Validation Results - Category: Top1={cat_results[0]:.4f}, Top5={cat_results[2]:.4f}")
        logging.info(f"Validation Results - Location: Top1={loc_results[0]:.4f}, Top5={loc_results[2]:.4f}")
    
    
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, combo_to_idx, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')
    return results
