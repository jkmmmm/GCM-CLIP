import torch

EPS = 1e-8


def flatten(grad_list):
    return torch.cat([g.reshape(-1) for g in grad_list])


def unflatten(flat, shapes):
    out = []
    offset = 0
    for shape in shapes:
        numel = shape.numel()
        out.append(flat[offset:offset + numel].view(shape))
        offset += numel
    return out


def combine(grads, method="gcm"):
    if method != "gcm":
        raise ValueError(
            f"Unsupported gradient combination method: {method!r}. "
            "Only the GCM gradient orthogonalization strategy (GOS) is available.")

    g_exp = grads["explicit"]
    g_imp = grads["implicit"]
    g_con = grads["contrastive"]
    proj = torch.dot(g_imp, g_exp) / (g_exp.norm() ** 2 + EPS)
    g_orth = g_imp - proj * g_exp
    return g_exp + g_orth + g_con
