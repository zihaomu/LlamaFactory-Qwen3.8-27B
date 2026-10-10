#!/usr/bin/env python3
"""Validate two-W7900 BF16 compute and RCCL collectives."""

from __future__ import annotations

import json
import os
from datetime import timedelta

import torch
import torch.distributed as distributed


WORLD_SIZE = 2
MATRIX_SIZE = 1024


def environment_integer(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing torchrun variable: {name}")
    return int(value)


def main() -> None:
    local_rank = environment_integer("LOCAL_RANK")
    rank = environment_integer("RANK")
    world_size = environment_integer("WORLD_SIZE")
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"Expected {WORLD_SIZE} ranks, got {world_size}")
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("The selected PyTorch build does not expose ROCm GPUs")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(f"Expected two visible GPUs, got {torch.cuda.device_count()}")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=3),
        device_id=device,
    )
    try:
        properties = torch.cuda.get_device_properties(device)
        architecture = str(getattr(properties, "gcnArchName", "unknown"))
        if not architecture.startswith("gfx1100"):
            raise RuntimeError(f"Expected W7900/gfx1100, got {architecture}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by the selected runtime")

        torch.manual_seed(20261009 + rank)
        value = torch.randn(
            MATRIX_SIZE,
            MATRIX_SIZE,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        loss = (value @ value.T).float().square().mean()
        loss.backward()
        reduced = torch.tensor(float(rank + 1), device=device)
        distributed.all_reduce(reduced)
        if reduced.item() != 3.0:
            raise RuntimeError(f"RCCL all_reduce returned {reduced.item()}, expected 3.0")
        if not torch.isfinite(loss) or value.grad is None or not torch.isfinite(value.grad).all():
            raise RuntimeError(f"Rank {rank} produced non-finite BF16 values")
        torch.cuda.synchronize(device)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "backend": distributed.get_backend(),
                    "device": properties.name,
                    "architecture": architecture,
                    "hip": torch.version.hip,
                    "loss": loss.item(),
                    "all_reduce_sum": reduced.item(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        distributed.barrier(device_ids=[local_rank])
        if rank == 0:
            print("DISTRIBUTED_SMOKE_PASS", flush=True)
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()