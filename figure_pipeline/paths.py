"""figure_pipeline: 重生成 Fig2a/2b/2d + Fig3a-e 的数据与图表。

所有脚本共享的路径/常量。检查点可通过环境变量 GCM_CKPT 覆盖（续训后换检查点重跑）。
"""
import os

PROJECT_ROOT = "/home/mmmmjk/gcm_clip"
VAD = "/home/mmmmjk/Virtual_Anatomical_Dataset"          # ForVA 主数据
SEG_DATA = "/home/mmmmjk/Seg_data"                        # 分割数据集 (13 类, 198 例, 真实阅片病例池)
OPEN_CLIP_SRC = os.path.join(PROJECT_ROOT, "biomedclip_finetuning", "open_clip", "src")

# ---------- 模型检查点 ----------
LOGS = os.path.join(OPEN_CLIP_SRC, "logs")
RUN_0807 = os.path.join(
    LOGS, "2026_08_07-12_49_56-model_hf-hub:microsoft-BiomedCLIP-PubMedBERT_256-vit_base_patch16_224-lr_0.001-b_176-j_4-p_amp")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

CKPT = {
    "gcm_clip": os.environ.get(
        "GCM_CKPT",
        os.path.join(RUN_0807, "checkpoints", "best_model.pt")),
    "biomedclip": os.path.join(MODEL_DIR, "biomedclip_model", "open_clip_pytorch_model.bin"),
    "pubmedclip": os.path.join(MODEL_DIR, "pubmedclip-vit-base-patch32"),
    "gloria": os.path.join(MODEL_DIR, "gloria", "pretrained", "Gloria_chexpert_resnet18.ckpt"),
    "mgca": os.path.join(MODEL_DIR, "MGCA", "pretrained", "mgca_vit_base.ckpt"),
    "medsiglip": os.path.join(MODEL_DIR, "medsiglip-448"),
    # 新基线（下载成功后生效）
    "biovil": os.path.join(MODEL_DIR, "biovil"),
    "chexzero": os.path.join(MODEL_DIR, "chexzero"),
    "medclip": os.path.join(MODEL_DIR, "medclip"),
}
HUB_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# 消融检查点（GOS 对比）
ABLATION_DIR = os.path.join(LOGS, "ablation")
ABLATIONS = ["vanilla", "gcm", "pcgrad", "cagrad", "graddrop"]

# ---------- 数据 ----------
VAD_DATA = os.path.join(VAD, "data")
FORENSIC_TRAIN_CSV = os.path.join(VAD_DATA, "forensic_CT_train.csv")            # UTF-8
FORENSIC_TEST_CSV = os.path.join(VAD_DATA, "balanced_samples_4096_method2.csv")  # UTF-8
FORENSIC_SUB_CSV = os.path.join(VAD_DATA, "forensic_sub_CT_train.csv")          # GBK
FORENSIC_STATS_JSON = os.path.join(VAD_DATA, "forensic_CT_statistics.json")     # GBK
CAT_LOC_MAPPING_JSON = os.path.join(VAD_DATA, "category_location_mapping.json") # GBK

GCM_DATA = os.path.join(PROJECT_ROOT, "data")
ROCO_CSV = os.path.join(GCM_DATA, "ROCOv2", "jpg_data", "selected_4096_category_balanced.csv")
ROCO_IMG_REMAP = ("/root/autodl-tmp/data/ROCOv2/jpg_data",
                  os.path.join(GCM_DATA, "ROCOv2", "jpg_data"))
MIMIC_DIR = os.path.join(GCM_DATA, "mimic_cxr")
PMC_IMG_DIR = os.path.join(GCM_DATA, "pmc_oa", "caption_T060_filtered_top4_sep_v0_subfigures")

# 旧服务器路径前缀重映射
PATH_REMAPS = [("/root/autodl-tmp/data/dataset_jpg", os.path.join(VAD, "dataset_jpg")),
               ROCO_IMG_REMAP]

# ---------- 输出 ----------
FIG_DATA = os.path.join(PROJECT_ROOT, "output", "figure_data")
MANIFESTS = os.path.join(FIG_DATA, "manifests")
CACHE = os.path.join(FIG_DATA, "cache")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "article_2", "figures")

# GCM_CLIP 审核重建的 verified 测试集 (见 verify_and_refill.py); 评估流水线直接读这些
# forva(2026-09-17): 旧 forva_test_4096_verified.csv 随旧服务器丢失且本地不可复现,
# 改指 build_train_val_split_v2.py 产出的 forva_val_4096_diverse.csv(40 类配额 +
# 嵌入多样性约束, 方法见 output/dataset_cleaning_for_reviewers.en.md) —— 与重训
# 主模型的 val 清单同源, 口径一致。
VERIFIED_MANIFESTS = {
    # GCM_FORVA_MANIFEST 可覆盖:用找回的原始 forva_test_4096_verified.csv(S9 同口径)
    "forva": os.environ.get("GCM_FORVA_MANIFEST",
                            os.path.join(MANIFESTS, "forva_val_4096_diverse.csv")),
    "roco": os.path.join(MANIFESTS, "roco_test_verified.csv"),
    "mimic": os.path.join(MANIFESTS, "mimic_test_1024_verified.csv"),
    "pmc": os.path.join(MANIFESTS, "pmc_test_4096_verified.csv"),
}

for _d in [MANIFESTS, CACHE, FIGURES_DIR,
           *[os.path.join(FIG_DATA, p) for p in
             ["fig2a", "fig2b", "fig2d", "fig3a", "fig3b", "fig3c", "fig3d", "fig3e",
              "fig4", "leakage_audit", "ablation"]]]:
    os.makedirs(_d, exist_ok=True)

# ---------- 模型展示顺序（ours 永远第一） ----------
DISPLAY_NAME = {
    "gcm_clip": "GCM_CLIP",
    "biomedclip": "BiomedCLIP",
    "pubmedclip": "PubMedCLIP",
    "gloria": "GLoRIA",
    "mgca": "MGCA",
    "medsiglip": "MedSigLIP",
    "biovil": "BioViL",
    "chexzero": "CheXzero",
    "medclip": "MedCLIP",
}
CORE_MODELS = ["gcm_clip", "biomedclip", "pubmedclip", "gloria", "mgca", "medsiglip"]
EXTRA_MODELS = ["biovil", "chexzero", "medclip"]      # 下载成功才加入


def model_list(include_extra: bool = True):
    """返回可用模型列表：core + 下载成功的 extra。"""
    models = list(CORE_MODELS)
    if include_extra:
        for m in EXTRA_MODELS:
            if m == "biovil" and os.path.isdir(CKPT[m]) and os.listdir(CKPT[m]):
                models.append(m)
            elif m in ("chexzero", "medclip") and os.path.exists(CKPT[m] + ".ok"):
                models.append(m)
    return models


def remap_path(p: str) -> str:
    for old, new in PATH_REMAPS:
        if p.startswith(old):
            return new + p[len(old):]
    return p


def read_json_gbk(path):
    import json
    with open(path, "r", encoding="gbk") as f:
        return json.load(f)


# ================================================================
# 消融 v2: 从 BiomedCLIP 初始化的重训矩阵 (logs/ablation_v2, 25 变体)
#   变体定义见 biomedclip_finetuning/open_clip/scripts/ablation_v2_variants.txt
# ================================================================
ABLATION_V2_DIR = os.path.join(LOGS, "ablation_v2")
ABLATIONS_V2 = [
    # A. 组件消融 (论文 Table S13: CLIP 基线 -> +Exp -> +Imp -> Exp+Imp -> +GOS)
    "s13_base", "s13_exp", "s13_imp", "s13_expimp", "s13_full",
    # B. GOS 梯度方法族 (全量组件, 仅换梯度手术方法)
    "gos_pcgrad", "gos_cagrad", "gos_graddrop", "gos_mgda",
    "gos_gradnorm", "gos_uncertain",
    # C. DSD 分解基变体 (全量组件+gcm, 仅换分解)
    "dsd_identity", "dsd_pca_global", "dsd_pca_noema", "dsd_random",
    "dsd_ica", "dsd_spca", "dsd_nmf", "dsd_gate", "dsd_m32",
    # D. ABC 聚类变体 (全量组件+gcm, 仅换聚类)
    "abc_plain", "abc_fixedk", "abc_gmm", "abc_hier", "abc_proto",
]
ABLATION_V2_FIG = os.path.join(FIG_DATA, "ablation_v2")
os.makedirs(ABLATION_V2_FIG, exist_ok=True)
