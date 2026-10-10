#!/usr/bin/env python3
"""Parse both public profiles through LLaMA-Factory in a two-rank context."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as distributed
import yaml
from llamafactory.hparams import get_train_args


EXPECTED = {
    "bf16-lora-r64-2x-w7900": {
        "rank": 64,
        "alpha": 128,
        "accumulation": 2,
        "quantization": None,
        "deepspeed": True,
    },
    "nf4-qlora-r128-1x-w7900": {
        "rank": 128,
        "alpha": 256,
        "accumulation": 4,
        "quantization": 4,
        "deepspeed": False,
    },
}


def main() -> None:
    project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
    local_rank = int(os.environ["LOCAL_RANK"])
    process_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

    for name, expected in EXPECTED.items():
        config_path = project_root / "configs" / "recommended" / f"{name}.yaml"
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_args, data_args, training_args, finetuning_args, _ = get_train_args(raw_config)

        if model_args.model_revision != os.environ["MODEL_REVISION"]:
            raise RuntimeError(f"{name} uses an unexpected model revision")
        if data_args.template != "qwen3_8" or data_args.cutoff_len != 512:
            raise RuntimeError(f"{name} uses unexpected data settings")
        if training_args.max_steps != 100 or not training_args.bf16 or training_args.fp16:
            raise RuntimeError(f"{name} uses unexpected training precision/steps")
        if training_args.per_device_train_batch_size != 1:
            raise RuntimeError(f"{name} uses an unexpected micro batch")
        if training_args.gradient_accumulation_steps != expected["accumulation"]:
            raise RuntimeError(f"{name} uses unexpected gradient accumulation")
        if finetuning_args.lora_rank != expected["rank"] or finetuning_args.lora_alpha != expected["alpha"]:
            raise RuntimeError(f"{name} uses unexpected LoRA rank/alpha")
        if finetuning_args.lora_target != ["all"]:
            raise RuntimeError(f"{name} does not target all linear modules")
        if not finetuning_args.freeze_vision_tower or not finetuning_args.freeze_multi_modal_projector:
            raise RuntimeError(f"{name} does not freeze multimodal modules")
        if model_args.quantization_bit != expected["quantization"]:
            raise RuntimeError(f"{name} uses unexpected quantization")
        if bool(training_args.deepspeed) != expected["deepspeed"]:
            raise RuntimeError(f"{name} uses unexpected DeepSpeed settings")
        if expected["quantization"] == 4:
            if model_args.quantization_type != "nf4" or not model_args.double_quantization:
                raise RuntimeError(f"{name} is not NF4 double-quantized")
        if process_rank == 0:
            print(
                f"CONFIG_PASS profile={name} rank={finetuning_args.lora_rank} "
                f"accumulation={training_args.gradient_accumulation_steps}"
            )

    if distributed.is_initialized():
        distributed.barrier(device_ids=[local_rank])
        distributed.destroy_process_group()
    if process_rank == 0:
        print("TRAINING_CONFIG_VALIDATION_PASS")


if __name__ == "__main__":
    main()