"""Minimal peft stub.

The open_clip fork imports `LoraConfig` / `get_peft_model` at module level, but no
LoRA code path is exercised during offline inference/figure generation. Raise only
if actually invoked, so a missing real peft install never blocks evaluation.
"""


class LoraConfig:
    def __init__(self, **kwargs):
        raise RuntimeError(
            "peft is not installed; LoRA fine-tuning is not available in this env")


def get_peft_model(*args, **kwargs):
    raise RuntimeError("peft is not installed; LoRA fine-tuning is not available")
