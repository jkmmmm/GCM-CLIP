import sys
import tempfile
import types
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import open_clip
from open_clip.model import GCMCLIP, DynamicSemanticDecoupling, \
    LEGACY_GCM_KEY_MAP, remap_legacy_gcm_state_dict
from open_clip_train import grad_methods

torch.manual_seed(0)

BATCH = 16
EMBED_DIM = 64
CONTEXT = 16
VOCAB = 512


def build_tiny_gcmclip():
    vision_cfg = dict(image_size=32, patch_size=16, width=64, head_width=16, layers=2, mlp_ratio=2.0)
    text_cfg = dict(context_length=CONTEXT, vocab_size=VOCAB, width=64, heads=4, layers=2, mlp_ratio=2.0)
    return GCMCLIP(
        embed_dim=EMBED_DIM,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        output_dict=True,
        implicit_start_epoch=0,
        dsd_components=8,
        dsd_ema_beta=0.9,
        dsd_norm_space=True,
        logit_scale_max=100.0,
    )


def make_batch():
    images = torch.randn(BATCH, 3, 32, 32)
    tokens = torch.randint(0, VOCAB, (BATCH, CONTEXT))
    labels = torch.zeros(BATCH, 4)
    labels[torch.arange(BATCH), torch.randint(0, 4, (BATCH,))] = 1.0
    return images, tokens, labels


def test_dynamic_semantic_decoupling():
    dsd = DynamicSemanticDecoupling(input_dim=EMBED_DIM, num_components=8, ema_beta=0.9)
    assert dsd.decoupling_initialized.item() == 0.0

    x = torch.randn(32, EMBED_DIM)
    out = dsd(x)
    assert out.shape == x.shape

    W, Winv, mean = dsd.compute_decoupling(x)
    assert W.shape == (8, EMBED_DIM) and Winv.shape == (8, EMBED_DIM) and mean.shape == (EMBED_DIM,)
    dsd.update_decoupling(W, Winv, mean)
    assert dsd.decoupling_initialized.item() == 1.0

    out = dsd(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    print("[PASS] DynamicSemanticDecoupling")


def test_gcmclip_forward_before_mining():
    model = build_tiny_gcmclip()
    model.eval()
    images, tokens, _ = make_batch()

    with torch.no_grad():
        out = model(images, tokens, epoch=1)

    assert out["image_features"].shape == (BATCH, EMBED_DIM)
    assert out["text_features"].shape == (BATCH, EMBED_DIM)
    assert out["implicit_category_0"].shape == (BATCH, 8)
    assert out["implicit_category_1"].shape == (BATCH, 16)
    assert out["implicit_category_0"].abs().sum() == 0
    assert out["logit_scale"].ndim == 0

    assert out["image_location_proj"].shape == (BATCH, EMBED_DIM // 4)
    assert out["image_health_proj"].shape == (BATCH, EMBED_DIM // 16)
    assert out["text_location_proj"].shape == (BATCH, EMBED_DIM // 4)
    assert out["text_health_proj"].shape == (BATCH, EMBED_DIM // 16)
    print("[PASS] GCMCLIP forward before implicit mining (gated, zeros branch)")


def test_gcmclip_mining_and_forward():
    model = build_tiny_gcmclip()
    model.eval()
    images, tokens, labels = make_batch()

    model.update_implicit_supervision(
        args=types.SimpleNamespace(accum_freq=1),
        epoch=1,
        text=[tokens],
        explicit_labels=labels,
    )
    assert model.dsd.decoupling_initialized.item() == 1.0
    assert model.coarse_centers.shape == (8, EMBED_DIM)
    assert model.fine_centers.shape == (16, EMBED_DIM)
    assert model.coarse_centers.abs().sum() > 0
    assert model.n_coarse_clusters == 8 and model.n_fine_clusters == 16

    with torch.no_grad():
        out = model(images, tokens, epoch=1)

    cat0, cat1 = out["implicit_category_0"], out["implicit_category_1"]
    assert cat0.shape == (BATCH, 8) and cat1.shape == (BATCH, 16)
    assert torch.allclose(cat0.sum(dim=-1), torch.ones(BATCH), atol=1e-4)
    assert torch.allclose(cat1.sum(dim=-1), torch.ones(BATCH), atol=1e-4)
    assert torch.isfinite(cat0).all() and torch.isfinite(cat1).all()

    with torch.no_grad():
        image_logits, text_logits = model.get_logits(images, tokens)
    assert image_logits.shape == (BATCH, BATCH) and text_logits.shape == (BATCH, BATCH)

    with torch.no_grad():
        out_repeated = model(images, tokens, epoch=1)
    assert torch.equal(out_repeated["implicit_category_0"], cat0)
    print("[PASS] GCMCLIP implicit mining + forward with implicit supervision")


def test_gcmclip_state_dict_roundtrip():
    model = build_tiny_gcmclip()
    images, tokens, labels = make_batch()
    model.update_implicit_supervision(
        args=types.SimpleNamespace(accum_freq=1), epoch=1, text=[tokens], explicit_labels=labels)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = str(Path(tmp) / "gcm_state.pt")
        torch.save(model.state_dict(), ckpt)
        restored = build_tiny_gcmclip()
        missing, unexpected = restored.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        assert not missing and not unexpected

    assert restored.dsd.decoupling_initialized.item() == 1.0
    assert torch.allclose(restored.coarse_centers, model.coarse_centers)
    restored.dsd.sanitize_after_load()
    assert restored.dsd.decoupling_initialized.item() == 1.0
    print("[PASS] GCMCLIP state_dict save/load roundtrip (strict) + sanitize_after_load")


def test_gcmclip_single_tower_forward():
    model = build_tiny_gcmclip()
    model.eval()
    images, _, _ = make_batch()

    with torch.no_grad():
        out = model(image=images, epoch=1)
    assert out["image_features"].shape == (BATCH, EMBED_DIM)
    assert out["implicit_category_0"] is None
    assert "text_location_proj" not in out and "text_health_proj" not in out
    assert out["image_location_proj"].shape == (BATCH, EMBED_DIM // 4)
    print("[PASS] GCMCLIP single-tower forward (zero_shot style)")


def test_factory_base_model():
    model = open_clip.create_model("ViT-B-32", output_dict=True)
    assert model.__class__.__name__ == "CLIP"
    assert hasattr(open_clip.factory, "GCMCLIP") and hasattr(open_clip.factory, "GCMCLIPLoss")
    assert hasattr(open_clip, "remap_legacy_gcm_state_dict")
    assert not hasattr(open_clip.model, "GCMCLIPClassifier")
    print("[PASS] factory import wiring + base CLIP construction")


def test_gcm_gradient_combination():
    torch.manual_seed(0)
    g_exp = torch.randn(97)
    g_imp = torch.randn(97)
    g_con = torch.randn(97)
    combined = grad_methods.combine({"explicit": g_exp, "implicit": g_imp, "contrastive": g_con}, "gcm")
    proj = torch.dot(g_imp, g_exp) / (g_exp.norm() ** 2 + 1e-8)
    expected = g_exp + (g_imp - proj * g_exp) + g_con
    assert torch.allclose(combined, expected, atol=1e-6)
    assert torch.allclose(combined, g_exp + g_con + g_imp - proj * g_exp, atol=1e-6)
    try:
        grad_methods.combine({"explicit": g_exp, "implicit": g_imp, "contrastive": g_con}, "pcgrad")
        raise AssertionError("expected ValueError for non-GCM method")
    except ValueError:
        pass
    flat = grad_methods.flatten([g_exp, g_imp])
    parts = grad_methods.unflatten(flat, [g_exp.shape, g_imp.shape])
    assert torch.equal(parts[0], g_exp) and torch.equal(parts[1], g_imp)
    print("[PASS] grad_methods: GCM/GOS orthogonal projection + flatten roundtrip")


def test_retrieval_metrics():
    from open_clip_train.train import get_clip_metrics
    torch.manual_seed(0)
    n = 32
    image_features = torch.nn.functional.normalize(torch.randn(n, 64), dim=-1)
    text_features = image_features.clone()
    metrics = get_clip_metrics(image_features, text_features, logit_scale=torch.tensor(100.0))
    assert metrics["image_to_text_R@1"] == 1.0 and metrics["text_to_image_R@1"] == 1.0
    assert metrics["image_to_text_mean_rank"] == 1.0
    assert 0.0 < metrics["image_to_text_R@5"] <= 1.0
    print("[PASS] cross-modal retrieval metrics (identity alignment R@1 == 1)")


def test_gcmclip_loss_forward():
    from open_clip.loss import GCMCLIPLoss
    model = train_reference_model()
    model.train()
    images, tokens, labels = make_batch()
    out = model(images, tokens, epoch=1)

    loss_fn = GCMCLIPLoss(
        local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1,
        use_horovod=False, implicit_start_epoch=0,
        loss_weights={'location': 1.0, 'health': 1.0,
                      'implicit_category0': 1.0, 'implicit_category1': 1.0})
    losses = loss_fn(
        out["image_features"], out["text_features"],
        image_location_proj=out["image_location_proj"],
        text_location_proj=out["text_location_proj"],
        image_health_proj=out["image_health_proj"],
        text_health_proj=out["text_health_proj"],
        location=labels, category=labels,
        implicit_category_0=out["implicit_category_0"],
        implicit_category_1=out["implicit_category_1"],
        epoch=1, logit_scale=out["logit_scale"].mean(),
        output_dict=True)
    expected_keys = {"contrastive_loss", "location_loss", "health_loss",
                     "implicit_category0_loss", "implicit_category1_loss"}
    assert expected_keys.issubset(set(losses)), sorted(losses)
    total = sum(losses.values())
    assert torch.isfinite(total), total
    total.backward()
    print("[PASS] GCMCLIPLoss multi-task forward + backward")


def train_reference_model():
    model = build_tiny_gcmclip()
    images, tokens, labels = make_batch()
    model.update_implicit_supervision(
        args=types.SimpleNamespace(accum_freq=1), epoch=1, text=[tokens], explicit_labels=labels)
    return model


def make_legacy_state_dict(model):
    new_to_old = {new: old for old, new in LEGACY_GCM_KEY_MAP.items()}
    return {new_to_old.get(k, k): v for k, v in model.state_dict().items()}


def test_legacy_checkpoint_remap():
    reference = train_reference_model()
    reference_state = reference.state_dict()

    legacy = make_legacy_state_dict(reference)
    for old_key in LEGACY_GCM_KEY_MAP:
        assert old_key in legacy, old_key
    for new_key in LEGACY_GCM_KEY_MAP.values():
        assert new_key not in legacy, new_key

    remapped = remap_legacy_gcm_state_dict(legacy)
    assert all(new_key in remapped for new_key in LEGACY_GCM_KEY_MAP.values())
    assert set(remapped) == set(reference_state)

    fresh = build_tiny_gcmclip()
    fresh.load_state_dict(remapped, strict=True)
    assert torch.equal(fresh.dsd.attribute_vectors, reference.dsd.attribute_vectors)
    assert torch.equal(fresh.dsd.reconstruction_operator, reference.dsd.reconstruction_operator)
    assert torch.equal(fresh.dsd.feature_mean, reference.dsd.feature_mean)
    assert torch.equal(fresh.dsd.decoupling_initialized, reference.dsd.decoupling_initialized)
    assert torch.equal(fresh.coarse_centers, reference.coarse_centers)
    assert torch.equal(fresh.fine_centers, reference.fine_centers)
    print("[PASS] remap_legacy_gcm_state_dict (strict load, tensor-identical)")

    idempotent = remap_legacy_gcm_state_dict(dict(reference_state))
    assert set(idempotent) == set(reference_state)
    print("[PASS] remap is a no-op on current-format state dicts")

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "legacy_gcm_clip.pt")
        torch.save(legacy, ckpt_path)
        resumed = build_tiny_gcmclip()
        incompatible = open_clip.factory.load_checkpoint(resumed, ckpt_path, strict=False)
        assert not incompatible.missing_keys, incompatible.missing_keys
        assert not incompatible.unexpected_keys, incompatible.unexpected_keys
        assert torch.equal(resumed.dsd.attribute_vectors, reference.dsd.attribute_vectors)
        assert torch.equal(resumed.coarse_centers, reference.coarse_centers)
        assert torch.equal(resumed.fine_centers, reference.fine_centers)
        assert resumed.dsd.decoupling_initialized.item() == 1.0
    print("[PASS] open_clip.factory.load_checkpoint resumes legacy checkpoints losslessly")

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "legacy_gcm_clip.pt")
        torch.save(legacy, ckpt_path)
        preloaded = open_clip.factory.load_state_dict(ckpt_path)
        remapped_dict = remap_legacy_gcm_state_dict(preloaded)
        target = build_tiny_gcmclip()
        model_state = target.state_dict()
        compatible = {k: v for k, v in remapped_dict.items()
                      if k in model_state and v.shape == model_state[k].shape}
        target.load_state_dict(compatible, strict=False)
        assert torch.equal(target.dsd.attribute_vectors, reference.dsd.attribute_vectors)
        assert torch.equal(target.fine_centers, reference.fine_centers)
    print("[PASS] create_model --pretrained path (shape-filtered load) recovers DSD state")


if __name__ == "__main__":
    test_dynamic_semantic_decoupling()
    test_gcmclip_forward_before_mining()
    test_gcmclip_mining_and_forward()
    test_gcmclip_state_dict_roundtrip()
    test_gcmclip_single_tower_forward()
    test_factory_base_model()
    test_legacy_checkpoint_remap()
    test_gcm_gradient_combination()
    test_retrieval_metrics()
    test_gcmclip_loss_forward()
    print("\nALL GCM SMOKE TESTS PASSED")
