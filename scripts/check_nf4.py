#!/usr/bin/env python3
"""Validate bitsandbytes NF4 forward/backward on one W7900."""

from __future__ import annotations

import json
import os
from importlib.metadata import version

import torch
from bitsandbytes.nn import Linear4bit, Params4bit


def main() -> None:
    if torch.version.hip is None:
        raise RuntimeError("The installed PyTorch build does not expose HIP")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")

    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", "unknown"))
    if not architecture.startswith("gfx1100"):
        raise RuntimeError(f"Expected W7900/gfx1100, got {architecture}")

    layer = Linear4bit(
        256,
        128,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    ).to("cuda")
    if not isinstance(layer.weight, Params4bit) or layer.weight.quant_state is None:
        raise RuntimeError("bitsandbytes did not create a quantized Params4bit weight")
    if layer.weight.data.dtype is not torch.uint8:
        raise RuntimeError(f"Unexpected NF4 storage dtype: {layer.weight.data.dtype}")

    inputs = torch.randn(4, 32, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    outputs = layer(inputs)
    loss = outputs.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(outputs).all() or inputs.grad is None or not torch.isfinite(inputs.grad).all():
        raise RuntimeError("NF4 forward/backward produced non-finite values")

    print(
        json.dumps(
            {
                "architecture": architecture,
                "bitsandbytes": version("bitsandbytes"),
                "device": properties.name,
                "hip": torch.version.hip,
                "loss": loss.item(),
                "quant_type": layer.weight.quant_state.quant_type,
                "nested": layer.weight.quant_state.nested,
                "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
            },
            sort_keys=True,
        )
    )
    print("NF4_SMOKE_PASS")


if __name__ == "__main__":
    main()