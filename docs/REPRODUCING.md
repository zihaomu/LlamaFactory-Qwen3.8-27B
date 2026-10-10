# Reproducing the profiles

## 1. Hardware and OS

The reference machine used Ubuntu 24.04, ROCm 7.2.4, Python 3.12, and two
Radeon PRO W7900D cards. GPU identifiers are physical host IDs and may differ
on another machine.

Before training, inspect available memory:

```bash
amd-smi metric --mem-usage --gpu all
```

Do not start the BF16 profile unless both selected cards are idle. BF16 rank 64
uses nearly all 48 GB on each card under the reference benchmark.

## 2. Create the pinned environment

```bash
git clone https://github.com/zihaomu/LlamaFactory-Qwen3.8-27B.git
cd LlamaFactory-Qwen3.8-27B
bash scripts/setup.sh
source scripts/env.sh
```

`setup.sh` performs the following operations inside the repository:

1. Creates `.venv` with Python 3.12.
2. Installs the pinned ROCm 7.2.4 PyTorch wheels.
3. Clones LLaMA-Factory into `src/LLaMA-Factory`.
4. Checks out commit `ce9dc9e072f80fa3abe0989d4ab90da25f083438`.
5. Installs the pinned training stack and editable LLaMA-Factory package.

The setup does not modify system ROCm packages.

## 3. Download the fixed model revision

```bash
bash scripts/download_model.sh
```

The default destination is `models/Qwen3.8-27B`. Override it when needed:

```bash
MODEL_PATH=/data/models/Qwen3.8-27B bash scripts/download_model.sh
```

If direct Hugging Face access is unavailable, set a mirror explicitly:

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
bash scripts/download_model.sh
```

## 4. Preflight checks

Two-GPU BF16/RCCL:

```bash
GPU_IDS=0,1 bash scripts/run_distributed_smoke.sh
```

One-GPU NF4 forward/backward:

```bash
HIP_VISIBLE_DEVICES=0 python scripts/check_nf4.py
```

Both commands must pass before starting the corresponding profile.

Parse both YAML profiles through the pinned LLaMA-Factory version:

```bash
HIP_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/validate_training_configs.py
```

## 5. Run the recommended profiles

Inspect the resolved commands without loading the model:

```bash
DRY_RUN=1 GPU_IDS=0,1 bash scripts/run_profile.sh bf16-r64
DRY_RUN=1 GPU_IDS=0 bash scripts/run_profile.sh nf4-r128
```

BF16 LoRA rank 64 on two cards:

```bash
GPU_IDS=0,1 bash scripts/run_profile.sh bf16-r64
```

NF4 QLoRA rank 128 on one card:

```bash
GPU_IDS=0 bash scripts/run_profile.sh nf4-r128
```

Each run creates an isolated directory under `outputs/<profile>/<run-id>` with
the effective config, command, environment versions, memory snapshots, log,
exit code, metrics, and adapter.

Validate the final adapter:

```bash
HIP_VISIBLE_DEVICES=0,1 python scripts/validate_adapter.py \
  --profile bf16-r64 \
  --run-dir outputs/bf16-r64/<run-id>

HIP_VISIBLE_DEVICES=0 python scripts/validate_adapter.py \
  --profile nf4-r128 \
  --run-dir outputs/nf4-r128/<run-id>
```

The BF16 validator requires the loaded base model to use both logical devices.

## 6. Use your own data

The committed `example_sft` dataset is deliberately small. Add an entry to
`data/dataset_info.json` following LLaMA-Factory's dataset format, then run:

```bash
DATASET=my_sft GPU_IDS=0 bash scripts/run_profile.sh nf4-r128
```

Keep these controls fixed when comparing profiles:

- train/eval/test split
- actual token length
- effective global batch
- optimizer steps or optimized token count
- random seed
- LoRA target set and dropout

## 7. Interpret the result correctly

- A successful synthetic run proves system compatibility, not task quality.
- NF4 stores base weights using a 4-bit codebook, while matrix compute remains
  BF16 and trainable/optimizer state may use FP32.
- DeepSpeed ZeRO-3 keeps BF16 base training feasible across two W7900 cards but
  does not provide a comfortable margin at rank 64.
- Do not infer that more GPUs improve QLoRA quality or speed. Under the tested
  global batch, the one-card NF4 path was faster.

## 8. Expected validation boundaries

```text
BF16 LoRA r64, 2 GPUs: 100-step PASS, adapter reload PASS
BF16 LoRA r128, 2 GPUs: OOM after step 25 evaluation
NF4 QLoRA r128, 1 GPU: 100-step PASS, adapter reload PASS
NF4 QLoRA r256, 1 GPU: one-step OOM
NF4 QLoRA r32, 2 GPUs: FLAKY; one NaN run, later exact 100-step PASS
```

If your result differs, compare the model revision, package versions, actual
sequence length, GPU visibility, and effective batch before changing multiple
parameters.

For dual-GPU NF4 experiments, use the fail-fast launcher documented in
[NF4_DDP_NAN_INVESTIGATION.md](NF4_DDP_NAN_INVESTIGATION.md) so a non-finite
gradient norm aborts before all adapter gradients are contaminated.