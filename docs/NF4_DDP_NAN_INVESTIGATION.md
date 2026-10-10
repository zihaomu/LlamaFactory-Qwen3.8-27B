# Dual-GPU NF4 rank-32 NaN investigation

## Status

The dual-GPU NF4 rank-32 result is **flaky**, not a deterministic failure and
not a confirmed upstream bug.

Observed outcomes under the same W7900, package, seed, data, and training
configuration:

```text
100-step run A: numerical FAIL from step 30
100-step run B: PASS, adapter reload PASS, fixed response = 42
```

The recommended NF4 profile remains one W7900 because it was faster at the same
global batch and has stronger repeated stability evidence.

## What failed in run A

- Loss was still finite at step 30: `2.367e-06`.
- Gradient norm first became NaN at step 30.
- Subsequent evaluation loss was NaN.
- All 992 saved LoRA tensors, containing 233,455,616 elements, were NaN.

This was a real parameter failure, not only a bad log value.

## Why every adapter tensor became NaN

The DDP training path clips gradients through:

```text
Transformers Trainer
  -> Accelerate clip_grad_norm_
  -> torch.nn.utils.clip_grad_norm_
```

PyTorch defaults to `error_if_nonfinite=False`. If one raw gradient or the total
norm becomes NaN, the clip coefficient is also NaN. The foreach multiply then
propagates that NaN coefficient to every gradient tensor. The fact that all 992
saved tensors were NaN therefore does not mean all modules failed independently.

The failed run did not capture gradients before clipping, so the first bad
operation cannot be recovered from that artifact.

## Reproduction work

Three targeted runs through step 31 did not reproduce the failure:

1. Instrumented run without evaluation.
2. Instrumented run with evaluation at step 25, matching the failed schedule.
3. Uninstrumented run with evaluation at step 25.

For both instrumented ranks at steps 28-31:

- all 233,455,616 gradient elements were finite before clipping;
- foreach and slow per-tensor norms were identical;
- step-30 norm was `0.000983523204922676`;
- clipping did not introduce non-finite values.

A later exact 100-step reproduction also passed:

| Metric | Value |
| --- | ---: |
| Completed steps | 100/100 |
| Final loss | `1.192e-07` |
| Final grad norm | `9.437e-06` |
| Final eval loss | `1.0217938495316048e-07` |
| Peak VRAM | 38,595 / 38,596 MB |
| Throughput | 342.528 tok/s |
| Adapter tensors | 992, all finite |
| Adapter reload | PASS, response `42` |

## Hardware checks

- No amdgpu reset, fault, timeout, or ring-hang event was present in the failed
  run's kernel-log window.
- ECC correctable, uncorrectable, and deferred counters were all zero on both
  W7900 cards.
- No retired, pending, or unresolved bad pages were reported.
- Failed and passed telemetry had similar VRAM, temperature, utilization, and
  power ranges.

These checks do not prove hardware can never be involved, but there is no
positive hardware-fault evidence.

## Upstream comparison

No issue was found that directly matches:

```text
W7900/gfx1100 + ROCm 7.2.4 + bitsandbytes NF4 + DDP + Qwen3.8
```

Nearby reports are not equivalent:

- [pytorch/pytorch#194004](https://github.com/pytorch/pytorch/issues/194004):
  ROCm 7.14, MI355X/gfx950, Qwen3 MoE, FSDP+TP+EP. Its maintainers report ROCm
  7.2 as stable and localize the regression to a different ROCm release.
- [pytorch/pytorch#126654](https://github.com/pytorch/pytorch/issues/126654):
  SDPA numerical behavior on Apple M1/macOS, not ROCm DDP or NF4.
- [unslothai/unsloth#3385](https://github.com/unslothai/unsloth/issues/3385):
  Gemma3 compiled-forward NaNs. This workload uses Qwen, no Unsloth, and no
  `torch.compile`.

Upstream involvement remains possible, especially for a rare asynchronous
kernel or distributed event, but the current evidence is not sufficient to
assign the bug to PyTorch, ROCm/RCCL, bitsandbytes, or LLaMA-Factory.

## Fail-fast mitigation

Use `scripts/train_with_finite_guard.py`. It forces gradient clipping to use:

```python
error_if_nonfinite=True
```

This does not fix the operation that creates the first NaN. It prevents a
non-finite norm from multiplying every gradient by NaN, stops immediately, and
avoids saving a silently poisoned adapter.

Example two-GPU NF4 rank-32 diagnostic run:

```bash
source scripts/env.sh
HIP_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/train_with_finite_guard.py \
  configs/recommended/nf4-qlora-r128-1x-w7900.yaml \
  lora_rank=32 \
  lora_alpha=64 \
  gradient_accumulation_steps=2 \
  output_dir=outputs/nf4-r32-g2-guarded
```

The guard passed a synthetic NaN unit test and a real 31-step dual-W7900 rank-32
run with step-25 evaluation.

## Recommendation

- Keep one-card NF4 as the default deployment/training profile.
- Treat two-card NF4 rank 32 as experimental and flaky, not impossible.
- Use fail-fast clipping and scan final adapter tensors for finite values.
- Require repeated successful runs on real data before calling it stable.
- File an upstream issue only after capturing the first non-finite tensor before
  clipping, including rank, step, operation, and a repeatable minimal script.
