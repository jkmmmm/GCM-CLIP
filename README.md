# GCM-CLIP 训练/评估代码与测试集数据包
## 目录结构

```
code_model/
├── README.md                          本说明
├── biomedclip_finetuning/open_clip/   GCM-CLIP 训练代码（open_clip 改造版）
│   ├── scripts/
│   │   └── run_gcm_clip_main_local.sh   主模型训练启动器（服务器口径, A800）
│   ├── src/
│   │   ├── open_clip/                   模型定义（model.py 含 GCM/DSD/ABC, loss.py 含
│   │   │                                GCMCLIPLoss 多任务监督对比 + GOS 梯度正交化）
│   │   └── open_clip_train/             训练入口（main/train/data/params/…,
│   │                                    仅保留训练 import 闭包内的模块）
│   ├── pyproject.toml / requirements*.txt / LICENSE
├── figure_pipeline/                   评估代码
│   ├── run_fig2b.py                     图文检索评估（I2T/T2I R@1/5/10 → Fig 2c）
│   ├── run_fig2a.py                     零样本分类评估（99 模板集成 + Dis/Loc/Joint）
│   ├── paths.py / loader.py / model_registry.py / verify_and_refill.py / _compat/
├── config/
│   └── forensic_CT_statistics.json    GCM 损失所需统计配置（--config; GBK 编码）
├── model/
│   └── best_model.pt                  GCM-CLIP 主模型权重（重训 run gcm_clip 的 best,
│                                      761MB, 超 GitHub 单文件限制不入仓库, 评审索取联系作者）
└── data/                              不入仓库（见"数据隐私"）, 本地保存/私下分发
    ├── forva_test_4096_verified.csv   论文测试集（SI Table S4 口径）
    └── dataset_jpg/…                  测试集引用的全部虚拟解剖影像（4,025 张）
```

## 训练（服务器口径）

入口 `python -m open_clip_train.main`，完整参数见 `scripts/run_gcm_clip_main_local.sh`：
## 评估

```bash
export PYTHONPATH=<repo>/biomedclip_finetuning/open_clip/src:<repo>
export GCM_CKPT=<repo>/model/best_model.pt
# 图文检索（I2T/T2I R@1/5/10）
python -m figure_pipeline.run_fig2b --models gcm_clip
# 零样本分类（Loc@1 / Dis@1 / Dis_Loc@5, 99 模板）
python -m figure_pipeline.run_fig2a --models gcm_clip [--quick] [--skip-robustness]
```

## 测试集与论文对照

`forva_test_4096_verified.csv`（= SI Table S4 数据源）:
- 4,096 图文对；40 类每类 62–106 对, 与 Table S4 逐类一致（40/40）
- 8 解剖区域（Abdomen/Chest/Head/Lower Leg/Neck/Pelvis/Unknown/Upper Leg）
- 16 死因场景；208 具尸体; 报告均值 37.1 词
- 注: 4,096 对中含 3 对同图重复行（4,093 张唯一影像）, 按论文"pairs"口径计数

**2026-10-02 数据修订**: 逐文件核验发现 34 个被引用影像为超长路径文件,
本地不可恢复; 按同类别替换后其文本与替代图不一致, 故连同同图对应行共
移除 68 行（cerebral_hemorrhage_death -62, coronary_heart_disease_death -6）。
当前版本: **4,028 图文对 / 4,025 张唯一影像**, 40 类齐全,
但 cerebral_hemorrhage_death 与 coronary_heart_disease_death 两类的逐类
计数与 SI Table S4 不再一致; 论文主表指标基于 4,096 口径。
移除行完整归档于 `_removed_rows.csv`（不入仓库）。

## 本地复现结果

| 指标 | 论文（Fig. 2c） | 本地复现 |
|---|---|---|
| Loc@1 | 0.724 | **0.7239** |
| Dis@1 | 0.676 | **0.6758** |
| Dis_Loc@5 | 0.782 | **0.7822** |
| I2T R@1 | 0.565 | **0.5654** |
| I2T R@5 | 0.806 | **0.8059** |
| I2T R@10 | 0.894 | **0.8936** |

全部指标与论文一致（差异 ≤0.0005, 为 per_sample 排序并列时的舍入）。
复现命令见上节; 结果文件写入 `output/figure_data/fig2a|fig2b/gcm_clip/`
（经 `/home/mmmmjk` 目录联结映射）。

## 数据隐私

`data/dataset_jpg/` 为虚拟解剖（post-mortem CT）影像, 已去标识化,
属敏感法医学案件材料。仅限论文评审/课题组内部使用,
不得公开分发（与稿件 Data Availability 声明一致）。
**因此 data/ 不入本仓库**, 影像与 CSV 仅本地保存或私下分发;
`model/best_model.pt` 权重同样不入仓库。
