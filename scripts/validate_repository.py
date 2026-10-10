#!/usr/bin/env python3
"""Static validation for the published configs, data, and result summary."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    for path in root.glob("**/*.json"):
        if ".git" not in path.parts:
            json.loads(path.read_text(encoding="utf-8"))

    bf16 = yaml.safe_load(
        (root / "configs" / "recommended" / "bf16-lora-r64-2x-w7900.yaml").read_text(
            encoding="utf-8"
        )
    )
    nf4 = yaml.safe_load(
        (root / "configs" / "recommended" / "nf4-qlora-r128-1x-w7900.yaml").read_text(
            encoding="utf-8"
        )
    )
    if (bf16["lora_rank"], bf16["lora_alpha"]) != (64, 128):
        raise RuntimeError("Unexpected BF16 LoRA rank/alpha")
    if bf16.get("quantization_bit") is not None or "deepspeed" not in bf16:
        raise RuntimeError("BF16 profile precision/distribution settings are invalid")
    if (nf4["lora_rank"], nf4["lora_alpha"]) != (128, 256):
        raise RuntimeError("Unexpected NF4 QLoRA rank/alpha")
    if nf4.get("quantization_bit") != 4 or nf4.get("quantization_type") != "nf4":
        raise RuntimeError("NF4 profile quantization settings are invalid")
    if not nf4.get("double_quantization") or "deepspeed" in nf4:
        raise RuntimeError("NF4 profile distribution settings are invalid")
    if 2 * bf16["gradient_accumulation_steps"] != 4:
        raise RuntimeError("BF16 profile effective global batch is not 4")
    if nf4["gradient_accumulation_steps"] != 4:
        raise RuntimeError("NF4 profile effective global batch is not 4")

    for config in (bf16, nf4):
        for field in ("model_name_or_path", "dataset_dir", "output_dir"):
            if Path(config[field]).is_absolute():
                raise RuntimeError(f"Published config uses an absolute {field}: {config[field]}")
        if config["cutoff_len"] != 512 or config["max_steps"] != 100:
            raise RuntimeError("Published profile uses unexpected benchmark controls")
        if config["save_strategy"] != "no" or not config["save_only_model"]:
            raise RuntimeError("Published profile must save only the final adapter")
        if not config["freeze_vision_tower"] or not config["freeze_multi_modal_projector"]:
            raise RuntimeError("Published profile does not freeze multimodal modules")

    summary = json.loads((root / "results" / "official-repeat2.json").read_text(encoding="utf-8"))
    profiles = {(item["path"], item["rank"]): item for item in summary["successful_profiles"]}
    if ("bf16-lora-g2", 64) not in profiles or ("nf4-qlora-g1", 128) not in profiles:
        raise RuntimeError("Headline profiles are missing from the result summary")
    if not all(item["adapter_reload"] for item in profiles.values()):
        raise RuntimeError("A successful result lacks adapter reload evidence")

    readme = (root / "README.md").read_text(encoding="utf-8")
    required = (
        "2 x W7900, BF16 LoRA, rank 64",
        "1 x W7900, NF4 QLoRA, rank 128",
        "The NF4 recommendation is one W7900, not two.",
    )
    if not all(text in readme for text in required):
        raise RuntimeError("README is missing a headline recommendation")
    print("REPOSITORY_VALIDATION_PASS")


if __name__ == "__main__":
    main()