# Measured results

## Test system

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B` |
| Model revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| GPU | AMD Radeon PRO W7900D, 49,136 MB, `gfx1100` |
| ROCm | 7.2.4 |
| PyTorch | 2.7.1 ROCm 7.2.4 build |
| LLaMA-Factory | 0.9.6.dev0 at `ce9dc9e072f80fa3abe0989d4ab90da25f083438` |
| DeepSpeed | 0.18.4 |
| bitsandbytes | 0.50.2 |

## Method

- Actual sequence length: 512 tokens
- LoRA targets: all discovered linear modules
- `lora_alpha = 2 * rank`
- Micro batch: 1 per GPU
- Effective global batch: 4
- Gradient checkpointing: enabled
- Attention: SDPA
- Compute dtype: BF16
- BF16 LoRA: two GPUs with DeepSpeed ZeRO-3
- NF4 QLoRA: bitsandbytes NF4 with double quantization
- P1: one optimizer step
- P2: 10 optimizer steps
- P3: 100 optimizer steps plus evaluation every 25 steps
- Every successful P3 cell was repeated; values below are from the second run
- Every successful second-run adapter was reloaded and returned `42` for the fixed inference sample

The train/eval sample was synthetic and looped back. Loss values are not model
quality measurements.

## Final matrix

| Rank | 2-GPU BF16 LoRA | 1-GPU NF4 QLoRA | 2-GPU NF4 QLoRA |
| ---: | --- | --- | --- |
| 8 | PASS; 47,797/47,723 MB; 263.2 tok/s | PASS; 22,834 MB; 356.9 tok/s | PASS; 34,448/38,029 MB; 340.5 tok/s |
| 16 | PASS; 46,509/46,435 MB; 251.9 tok/s | PASS; 24,728 MB; 357.9 tok/s | PASS; 34,691/34,692 MB; 345.6 tok/s |
| 32 | PASS; 49,107/49,077 MB; 250.4 tok/s | PASS; 25,535 MB; 355.3 tok/s | FLAKY; one NaN run, later 100-step PASS; 342.5 tok/s |
| 64 | PASS; 48,791/48,763 MB; 247.8 tok/s | PASS; 29,504 MB; 353.3 tok/s | STOP; rank-32 gate failed |
| 128 | OOM after step 25 evaluation | PASS; 38,324 MB; 347.1 tok/s | STOP; rank-32 gate failed |
| 256 | P1 OOM | P1 OOM | P1 OOM |

Single-GPU BF16 LoRA is not in the matrix because the approximately 54 GB
BF16 base weights exceed one 48 GB card without offload.

## What to use

### BF16 is required

Use **two W7900 cards, LoRA rank 64, and DeepSpeed ZeRO-3**.

This is the highest power-of-two rank that passed two 100-step runs and adapter
reload. It is a capacity-edge configuration: the second run peaked at
48,791/48,763 MB. For unattended or variable-length workloads, rank 16 or a
smaller target set is safer.

### A single card is preferred

Use **one W7900 card, NF4 QLoRA rank 128**.

The second 100-step run achieved 347.1 tok/s and peaked at 38,324 MB. This was
the highest power-of-two NF4 rank that passed the full test; rank 256 failed P1.

### Why not two-card NF4?

At the same effective global batch, dual-card NF4 did not outperform the
single-card run. One rank-32 run produced NaN gradient norms and NaN evaluation
loss, but a later exact 100-step reproduction passed and returned `42` after
adapter reload. The result is therefore flaky and unresolved, not a
deterministic rank-32 failure. We still do not present two-card NF4 as the
default path because it was slower and has weaker stability evidence.

See [NF4_DDP_NAN_INVESTIGATION.md](NF4_DDP_NAN_INVESTIGATION.md) for the raw
evidence, upstream comparison, and fail-fast mitigation.

## Evidence status

```text
P1: capacity matrix completed
P2: 15/15 candidate cells completed 10 steps
P3 first run: 11 PASS, 1 FAIL, 1 OOM, 2 STOP
P3 second run: 11/11 successful cells passed again
Adapter reload: 11/11 passed, fixed response = 42
Dual-GPU NF4 r32 follow-up: exact 100-step PASS, adapter reload = 42
```

The raw 56 GB output directory is intentionally not committed. The repository
contains the configs, exact version pins, scripts, and summarized evidence.