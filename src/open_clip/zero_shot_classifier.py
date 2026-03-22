from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
import itertools

def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def build_zero_shot_classifier(
        model,
        tokenizer,
        classnames_0,  # 第一类分类（如类别）
        classnames_1,  # 第二类分类（如位置）
        templates_joint: Sequence[Union[Callable, str]],  # 联合分类模板（同时包含两类）
        templates_0: Sequence[Union[Callable, str]],      # 第一类单独分类模板
        templates_1: Sequence[Union[Callable, str]],      # 第二类单独分类模板
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """构建零样本分类器权重，同时生成联合分类器和两个独立分类器
    
    Args:
        model: CLIP模型实例
        tokenizer: CLIP分词器实例
        classnames_0: 第一类（如类别）名称列表
        classnames_1: 第二类（如位置）名称列表
        templates_joint: 联合分类的模板（需包含{f}和{b}）
        templates_0: 第一类单独分类的模板（需包含{class}）
        templates_1: 第二类单独分类的模板（需包含{class}）
        num_classes_per_batch: 每批处理的类别数
        device: 运行设备
        use_tqdm: 是否使用进度条
    """
    # 验证输入有效性
    assert isinstance(templates_joint, Sequence) and len(templates_joint) > 0
    assert isinstance(templates_0, Sequence) and len(templates_0) > 0
    assert isinstance(templates_1, Sequence) and len(templates_1) > 0
    assert len(classnames_0) > 0, "classnames_0不能为空"
    assert len(classnames_1) > 0, "classnames_1不能为空"
    
    # 验证模板格式
    if isinstance(templates_joint[0], str):
        assert "{f}" in templates_joint[0] and "{b}" in templates_joint[0], \
            "联合模板必须包含 {f} 和 {b}"

    
    # 创建各类别映射字典
    class_combinations = list(itertools.product(classnames_0, classnames_1))
    combo_to_idx = {combo: idx for idx, combo in enumerate(class_combinations)}
    cat_to_idx = {cat: idx for idx, cat in enumerate(classnames_0)}
    loc_to_idx = {loc: idx for idx, loc in enumerate(classnames_1)}
    
    # 进度条包装器
    if use_tqdm:
        import tqdm
        iter_wrap = lambda x, total: tqdm.tqdm(x, total=total)
    else:
        iter_wrap = lambda x, total: x
    
    def _process_batch(batch_classes, templates, is_joint=False):
        """处理批次类别并生成嵌入"""
        num_batch = len(batch_classes)
        num_templates = len(templates)
        texts = []
        # 生成文本模板
        for item in batch_classes:
            for tpl in templates:
                if is_joint:
                    texts.append(tpl.format(f=item[0], b=item[1]))
                else:
                    texts.append(tpl.format(c=item))
        
        # 编码文本
        texts = tokenizer(texts).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(texts, normalize=True)
        
        # 按模板平均并归一化
        embeddings = embeddings.reshape(num_batch, num_templates, -1).mean(dim=1)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        return embeddings.T  # 转置为 [特征维度, 类别数]
    
    # 1. 构建联合分类器 (category + location)
    num_joint = len(class_combinations)
    if num_classes_per_batch:
        joint_batches = list(batched(class_combinations, num_classes_per_batch))
        joint_embeds = [_process_batch(batch, templates_joint, is_joint=True) 
                       for batch in iter_wrap(joint_batches, len(joint_batches))]
        joint_classifier = torch.cat(joint_embeds, dim=1)
    else:
        joint_classifier = _process_batch(class_combinations, templates_joint, is_joint=True)
    
    # 2. 构建类别单独分类器 (仅category)
    num_cat = len(classnames_0)
    if num_classes_per_batch:
        cat_batches = list(batched(classnames_0, num_classes_per_batch))
        cat_embeds = [_process_batch(batch, templates_0)
                    for batch in iter_wrap(cat_batches, len(cat_batches))]
        cat_classifier = torch.cat(cat_embeds, dim=1)
    else:
        cat_classifier = _process_batch(classnames_0, templates_0)
    
    # 3. 构建位置单独分类器 (仅location)
    num_loc = len(classnames_1)
    if num_classes_per_batch:
        loc_batches = list(batched(classnames_1, num_classes_per_batch))
        loc_embeds = [_process_batch(batch, templates_1)
                    for batch in iter_wrap(loc_batches, len(loc_batches))]
        loc_classifier = torch.cat(loc_embeds, dim=1)
    else:
        loc_classifier = _process_batch(classnames_1, templates_1)
    
    return (joint_classifier, combo_to_idx,
            cat_classifier, cat_to_idx,
            loc_classifier, loc_to_idx)

def build_zero_shot_classifier_legacy(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names 1 by 1
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    if use_tqdm:
        import tqdm
        iter_wrap = tqdm.tqdm
    else:
        iter_wrap = iter

    use_format = isinstance(templates[0], str)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in iter_wrap(classnames):
            texts = [template.format(classname) if use_format else template(classname) for template in templates]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights

