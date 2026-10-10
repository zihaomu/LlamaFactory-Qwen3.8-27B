#!/usr/bin/env python3
"""Run LLaMA-Factory with fail-fast non-finite gradient clipping."""

from __future__ import annotations

import torch
from llamafactory.train.tuner import run_exp


_original_clip_grad_norm = torch.nn.utils.clip_grad_norm_


def guarded_clip_grad_norm(
    parameters,
    max_norm,
    norm_type=2.0,
    error_if_nonfinite=False,
    foreach=None,
):
    del error_if_nonfinite
    return _original_clip_grad_norm(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=True,
        foreach=foreach,
    )


def main() -> None:
    torch.nn.utils.clip_grad_norm_ = guarded_clip_grad_norm
    run_exp()


if __name__ == "__main__":
    main()
