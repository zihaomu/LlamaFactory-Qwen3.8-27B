#!/usr/bin/env python3
"""Reload a recommended profile adapter and run a fixed inference sample."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from llamafactory.chat import ChatModel


PROFILES = {
    "bf16-r64": {"rank": 64, "method": "lora", "gpus": 2},
    "nf4-r128": {"rank": 128, "method": "qlora", "gpus": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(Path(os.environ["OUTPUT_ROOT"]).resolve()):
        raise RuntimeError(f"Run directory is outside OUTPUT_ROOT: {run_dir}")
    adapter_dir = run_dir / "adapter"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise RuntimeError(f"Adapter weights are missing: {adapter_path}")

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    rank = int(adapter_config["r"])
    alpha = int(adapter_config["lora_alpha"])
    if rank != profile["rank"] or alpha != 2 * rank:
        raise RuntimeError(f"Unexpected adapter r/alpha: {rank}/{alpha}")
    if torch.cuda.device_count() != profile["gpus"]:
        raise RuntimeError(f"Expected {profile['gpus']} visible GPU(s), got {torch.cuda.device_count()}")

    model_args: dict[str, object] = {
        "model_name_or_path": os.environ["MODEL_PATH"],
        "model_revision": os.environ["MODEL_REVISION"],
        "adapter_name_or_path": str(adapter_dir),
        "template": os.environ["MODEL_TEMPLATE"],
        "finetuning_type": "lora",
        "infer_backend": "huggingface",
        "trust_remote_code": False,
        "flash_attn": "sdpa",
        "enable_thinking": False,
    }
    if profile["method"] == "qlora":
        model_args.update(
            {
                "quantization_bit": 4,
                "quantization_method": "bnb",
                "quantization_type": "nf4",
                "double_quantization": True,
            }
        )

    torch.cuda.reset_peak_memory_stats()
    chat_model = ChatModel(model_args)
    raw_device_map = getattr(chat_model.engine.model, "hf_device_map", {})
    device_map = {name: str(device) for name, device in raw_device_map.items()}
    if profile["method"] == "lora":
        assigned_devices = set(device_map.values())
        if not ({"0", "1"} <= assigned_devices or {"cuda:0", "cuda:1"} <= assigned_devices):
            raise RuntimeError(f"BF16 base model was not split across both GPUs: {device_map}")

    messages = [{"role": "user", "content": "Add 17 and 25. Return only the integer result."}]
    responses = chat_model.chat(
        messages,
        max_new_tokens=16,
        do_sample=False,
        skip_special_tokens=True,
    )
    if len(responses) != 1 or not responses[0].response_text.strip():
        raise RuntimeError(f"Invalid response: {responses}")
    torch.cuda.synchronize()

    record = {
        "profile": args.profile,
        "rank": rank,
        "alpha": alpha,
        "adapter_path": str(adapter_dir),
        "device_map": device_map,
        "messages": messages,
        "response": asdict(responses[0]),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
    }
    (run_dir / "validation_samples.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    print("ADAPTER_RELOAD_INFERENCE_PASS")


if __name__ == "__main__":
    main()