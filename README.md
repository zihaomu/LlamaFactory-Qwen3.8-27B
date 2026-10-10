# Qwen3.8-27B LoRA/QLoRA on AMD W7900

Reproducible LLaMA-Factory recipes for fine-tuning
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) on AMD Radeon PRO W7900 GPUs.

The repository records measurements from ROCm 7.2.4, two 48 GB W7900 cards,
and a fixed Qwen3.8-27B revision. It includes portable configurations,
environment pins, smoke tests, and the measured rank matrix.

## Recommended profiles

| Requirement | Use this profile | Validated result |
| --- | --- | --- |
| BF16 base weights | **2 x W7900, BF16 LoRA, rank 64, ZeRO-3** | Two 100-step runs and adapter reload passed |
| Lowest hardware requirement | **1 x W7900, NF4 QLoRA, rank 128** | Two 100-step runs and adapter reload passed |

The short answer is:

```text
Need BF16?       2 x W7900, LoRA rank 64, DeepSpeed ZeRO-3
Need 4-bit NF4?  1 x W7900, QLoRA rank 128
```

**The NF4 recommendation is one W7900, not two.** We tested two-card NF4
training as well. One rank-32 run produced non-finite gradients, while a later
exact 100-step reproduction passed and its adapter reloaded successfully. We
therefore classify it as flaky, not a deterministic rank-32 failure. With the
same global batch size, two-card NF4 was also slower than the single-card path
in this microbenchmark.

## 中文结论

- 如果必须使用 BF16 基座：使用 **两张 W7900 + LoRA rank 64 + ZeRO-3**。
- 如果希望降低硬件要求：使用 **一张 W7900 + NF4 QLoRA rank 128**。
- NF4 QLoRA 的推荐方案是单卡，不是双卡。双卡 NF4 rank 32 一次出现 NaN，后续同条件百步复现通过，因此属于 flaky，而不是必现失败。
- 上述 rank 是完成系统验证的最大 2 倍档位，不代表真实业务质量一定最优。

### Important limits

- BF16 rank 64 is the largest power-of-two rank that completed the 100-step
  benchmark twice. It used 48,791 / 48,763 MB in the official repeat, leaving
  almost no VRAM safety margin. Use rank 16 or reduce sequence/targets when
  operational headroom matters.
- Single-card NF4 rank 128 completed both 100-step runs at 38,324 MB peak VRAM.
  Rank 256 failed the one-step capacity probe.
- "Recommended" here means the best validated **system profile** under the
  included synthetic workload. It does not establish the best model quality.
  Choose the final rank with a real training set and an independent test set.

## Measured headline results

Official numbers below use the second 100-step run, actual 512-token input,
micro batch 1, effective global batch 4, gradient checkpointing, SDPA, and
`lora_alpha = 2 * rank`.

| Profile | Rank | GPUs | Peak VRAM | Throughput | Adapter reload |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 LoRA + ZeRO-3 | 64 | 2 | 48,791 / 48,763 MB | 247.8 tok/s | PASS, output `42` |
| NF4 QLoRA | 128 | 1 | 38,324 MB | 347.1 tok/s | PASS, output `42` |

See [docs/RESULTS.md](docs/RESULTS.md) for the full matrix and failure
boundaries.

The rank-32 NaN investigation is documented in
[docs/NF4_DDP_NAN_INVESTIGATION.md](docs/NF4_DDP_NAN_INVESTIGATION.md).

## Quick start

Prerequisites:

- Ubuntu 24.04 or a compatible Linux distribution
- ROCm 7.2.4
- Python 3.12
- One or two Radeon PRO W7900 (`gfx1100`) GPUs
- `amd-smi`, `git`, and enough disk space for Qwen3.8-27B

```bash
git clone https://github.com/zihaomu/LlamaFactory-Qwen3.8-27B.git
cd LlamaFactory-Qwen3.8-27B

bash scripts/setup.sh
source scripts/env.sh
bash scripts/download_model.sh
```

Run the BF16 rank-64 profile on two physical GPUs:

```bash
GPU_IDS=0,1 bash scripts/run_profile.sh bf16-r64
```

Run the NF4 rank-128 profile on one physical GPU:

```bash
GPU_IDS=0 bash scripts/run_profile.sh nf4-r128
```

After a run completes, reload the final adapter and execute the fixed inference
check. Replace `<run-id>` with the directory printed by `run_profile.sh`:

```bash
HIP_VISIBLE_DEVICES=0,1 python scripts/validate_adapter.py \
  --profile bf16-r64 \
  --run-dir outputs/bf16-r64/<run-id>

HIP_VISIBLE_DEVICES=0 python scripts/validate_adapter.py \
  --profile nf4-r128 \
  --run-dir outputs/nf4-r128/<run-id>
```

The included dataset is synthetic and intended only for a reproducibility
smoke run. Register your own dataset in `data/dataset_info.json`, then override
the dataset without editing the committed config:

```bash
DATASET=my_sft GPU_IDS=0 bash scripts/run_profile.sh nf4-r128
```

Other useful overrides:

```bash
MODEL_PATH=/path/to/Qwen3.8-27B \
OUTPUT_ROOT=/path/to/outputs \
GPU_IDS=0,1 \
bash scripts/run_profile.sh bf16-r64
```

## Repository layout

```text
configs/
  deepspeed/ds_z3_config.json
  recommended/bf16-lora-r64-2x-w7900.yaml
  recommended/nf4-qlora-r128-1x-w7900.yaml
data/
  dataset_info.json
  example_sft.json
docs/
  REPRODUCING.md
  RESULTS.md
requirements/
scripts/
results/
versions.env
```

## Reproducibility scope

The measured matrix used one synthetic 512-token train/eval loopback sample.
Its loss is not a quality metric. The repository publishes the resource and
stability result, not model weights, adapters, caches, or the 56 GB raw run
directory.

For exact setup steps and validation gates, read
[docs/REPRODUCING.md](docs/REPRODUCING.md).

## License and upstream projects

This repository is licensed under Apache-2.0. LLaMA-Factory and Qwen model
assets retain their own upstream licenses and terms.