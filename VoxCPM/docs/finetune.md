# VoxCPM Fine-tuning Guide

This guide covers how to fine-tune VoxCPM models with two approaches: full fine-tuning and LoRA fine-tuning.

### 🎓 SFT (Supervised Fine-Tuning)

Full fine-tuning updates all model parameters. Suitable for:
- 📊 Large, specialized datasets
- 🔄 Cases where significant behavior changes are needed

### ⚡ LoRA Fine-tuning

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:
- 🎯 Trains only a small number of additional parameters
- 💾 Significantly reduces memory requirements and training time
- 🔀 Supports multiple LoRA adapters with hot-swapping



## Table of Contents

- [Quick Start: WebUI](#quick-start-webui)
- [Data Preparation](#data-preparation)
- [Full Fine-tuning](#full-fine-tuning)
- [LoRA Fine-tuning](#lora-fine-tuning)
- [Inference](#inference)
- [LoRA Hot-swapping](#lora-hot-swapping)
- [FAQ](#faq)

---

## Quick Start: WebUI

For users who prefer a graphical interface, we provide `lora_ft_webui.py` - a comprehensive WebUI for training and inference:

### Launch WebUI

```bash
python lora_ft_webui.py
```

Then open `http://localhost:7860` in your browser.

### Features

- **🚀 Training Tab**: Configure and start LoRA training with an intuitive interface
  - Set training parameters (learning rate, batch size, LoRA rank, etc.)
  - Monitor training progress in real-time
  - Resume training from existing checkpoints

- **🎵 Inference Tab**: Generate audio with trained models
  - Automatic base model loading from LoRA checkpoint config
  - Voice cloning with automatic ASR (reference text recognition)
  - Hot-swap between multiple LoRA models
  - Zero-shot TTS without reference audio

## Data Preparation

Training data should be prepared as a JSONL manifest file, with one sample per line:

```jsonl
{"audio": "path/to/audio1.wav", "text": "Transcript of audio 1."}
{"audio": "path/to/audio2.wav", "text": "Transcript of audio 2."}
{"audio": "path/to/audio3.wav", "text": "Optional duration field.", "duration": 3.5}
{"audio": "path/to/audio4.wav", "text": "Optional dataset_id for multi-dataset.", "dataset_id": 1}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `audio` | Path to audio file (absolute or relative) |
| `text` | Corresponding transcript |

### Optional Fields

| Field | Description |
|-------|-------------|
| `duration` | Audio duration in seconds (speeds up sample filtering) |
| `dataset_id` | Dataset ID for multi-dataset training (default: 0) |

### Requirements

- Audio format: WAV
- Sample rate: 16kHz for VoxCPM-0.5B, 44.1kHz for VoxCPM1.5
- Text: Transcript matching the audio content

See `examples/train_data_example.jsonl` for a complete example.

---

## Full Fine-tuning

Full fine-tuning updates all model parameters. Suitable for large datasets or when significant behavior changes are needed.

### Configuration

Create `conf/voxcpm_v1.5/voxcpm_finetune_all.yaml`:

```yaml
pretrained_path: /path/to/VoxCPM1.5/
train_manifest: /path/to/train.jsonl
val_manifest: ""

sample_rate: 44100
batch_size: 16
grad_accum_steps: 1
num_workers: 2
num_iters: 2000
log_interval: 10
valid_interval: 1000
save_interval: 1000

learning_rate: 0.00001   # Use smaller LR for full fine-tuning
weight_decay: 0.01
warmup_steps: 100
max_steps: 2000
max_batch_tokens: 8192

save_path: /path/to/checkpoints/finetune_all
tensorboard: /path/to/logs/finetune_all

lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
```

### Training

```bash
# Single GPU
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v1.5/voxcpm_finetune_all.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v1.5/voxcpm_finetune_all.yaml
```

### Checkpoint Structure

Full fine-tuning saves a complete model directory that can be loaded directly:

```
checkpoints/finetune_all/
└── step_0002000/
    ├── model.safetensors     # Model weights (excluding audio_vae)
    ├── config.json            # Model config
    ├── audiovae.pth           # Audio VAE weights
    ├── tokenizer.json         # Tokenizer
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── optimizer.pth
    └── scheduler.pth
```

---

## LoRA Fine-tuning

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that trains only a small number of additional parameters, significantly reducing memory requirements.

### Configuration

Create `conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml`:

```yaml
pretrained_path: /path/to/VoxCPM1.5/
train_manifest: /path/to/train.jsonl
val_manifest: ""

sample_rate: 44100
batch_size: 16
grad_accum_steps: 1
num_workers: 2
num_iters: 2000
log_interval: 10
valid_interval: 1000
save_interval: 1000

learning_rate: 0.0001    # LoRA can use larger LR
weight_decay: 0.01
warmup_steps: 100
max_steps: 2000
max_batch_tokens: 8192

save_path: /path/to/checkpoints/finetune_lora
tensorboard: /path/to/logs/finetune_lora

lambdas:
  loss/diff: 1.0
  loss/stop: 1.0

# LoRA configuration
lora:
  enable_lm: true        # Apply LoRA to Language Model
  enable_dit: true       # Apply LoRA to Diffusion Transformer
  enable_proj: false     # Apply LoRA to projection layers (optional)
  
  r: 32                  # LoRA rank (higher = more capacity)
  alpha: 16              # LoRA alpha, scaling = alpha / r
  dropout: 0.0
  
  # Target modules
  target_modules_lm: ["q_proj", "v_proj", "k_proj", "o_proj"]
  target_modules_dit: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Distribution options (optional)
# hf_model_id: "openbmb/VoxCPM1.5"  # HuggingFace ID
# distribute: true                   # If true, save hf_model_id in lora_config.json
```

### LoRA Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `enable_lm` | Apply LoRA to LM (language model) | `true` |
| `enable_dit` | Apply LoRA to DiT (diffusion model) | `true` (required for voice cloning) |
| `r` | LoRA rank (higher = more capacity) | 16-64 |
| `alpha` | Scaling factor, `scaling = alpha / r` | Usually `r/2` or `r` |
| `target_modules_*` | Layer names to add LoRA | attention layers |

### Distribution Options (Optional)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hf_model_id` | HuggingFace model ID (e.g., `openbmb/VoxCPM1.5`) | `""` |
| `distribute` | If `true`, save `hf_model_id` as `base_model` in checkpoint; otherwise save local `pretrained_path` | `false` |

> **Note**: If `distribute: true`, `hf_model_id` is required.

### Training

```bash
# Single GPU
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v1.5/voxcpm_finetune_lora.yaml
```

### Checkpoint Structure

LoRA training saves LoRA parameters and configuration:

```
checkpoints/finetune_lora/
└── step_0002000/
    ├── lora_weights.safetensors    # Only lora_A, lora_B parameters
    ├── lora_config.json            # LoRA config + base model path
    ├── optimizer.pth
    └── scheduler.pth
```

The `lora_config.json` contains:
```json
{
  "base_model": "/path/to/VoxCPM1.5/",
  "lora_config": {
    "enable_lm": true,
    "enable_dit": true,
    "r": 32,
    "alpha": 16,
    ...
  }
}
```

The `base_model` field contains:
- Local path (default): when `distribute: false` or not set
- HuggingFace ID: when `distribute: true` (e.g., `"openbmb/VoxCPM1.5"`)

This allows loading LoRA checkpoints without the original training config file.

---

## Inference

### Full Fine-tuning Inference

The checkpoint directory is a complete model, load it directly:

```bash
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir /path/to/checkpoints/finetune_all/step_0002000 \
    --text "Hello, this is the fine-tuned model." \
    --output output.wav
```

With voice cloning:

```bash
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir /path/to/checkpoints/finetune_all/step_0002000 \
    --text "This is voice cloning result." \
    --prompt_audio /path/to/reference.wav \
    --prompt_text "Reference audio transcript" \
    --output cloned_output.wav
```

### LoRA Inference

LoRA inference only requires the checkpoint directory (base model path and LoRA config are read from `lora_config.json`):

```bash
python scripts/test_voxcpm_lora_infer.py \
    --lora_ckpt /path/to/checkpoints/finetune_lora/step_0002000 \
    --text "Hello, this is LoRA fine-tuned result." \
    --output lora_output.wav
```

With voice cloning:

```bash
python scripts/test_voxcpm_lora_infer.py \
    --lora_ckpt /path/to/checkpoints/finetune_lora/step_0002000 \
    --text "This is voice cloning with LoRA." \
    --prompt_audio /path/to/reference.wav \
    --prompt_text "Reference audio transcript" \
    --output cloned_output.wav
```

Override base model path (optional):

```bash
python scripts/test_voxcpm_lora_infer.py \
    --lora_ckpt /path/to/checkpoints/finetune_lora/step_0002000 \
    --base_model /path/to/another/VoxCPM1.5 \
    --text "Use different base model." \
    --output output.wav
```

---

## LoRA Hot-swapping

LoRA supports dynamic loading, unloading, and switching at inference time without reloading the entire model.

### API Reference

```python
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# 1. Load model with LoRA structure and weights
lora_cfg = LoRAConfig(
    enable_lm=True, 
    enable_dit=True, 
    r=32, 
    alpha=16,
    target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
    target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
)
model = VoxCPM.from_pretrained(
    hf_model_id="openbmb/VoxCPM1.5",  # or local path
    load_denoiser=False,              # Optional: disable denoiser for faster loading
    optimize=True,                    # Enable torch.compile acceleration
    lora_config=lora_cfg,
    lora_weights_path="/path/to/lora_checkpoint",
)

# 2. Generate audio
audio = model.generate(
    text="Hello, this is LoRA fine-tuned result.",
    prompt_wav_path="/path/to/reference.wav",  # Optional: for voice cloning
    prompt_text="Reference audio transcript",   # Optional: for voice cloning
)

# 3. Disable LoRA (use base model only)
model.set_lora_enabled(False)

# 4. Re-enable LoRA
model.set_lora_enabled(True)

# 5. Unload LoRA (reset weights to zero)
model.unload_lora()

# 6. Hot-swap to another LoRA
loaded, skipped = model.load_lora("/path/to/another_lora_checkpoint")
print(f"Loaded {len(loaded)} params, skipped {len(skipped)}")

# 7. Get current LoRA weights
lora_state = model.get_lora_state_dict()
```

### Simplified Usage (Load from lora_config.json)

If your checkpoint contains `lora_config.json` (saved by the training script), you can load everything automatically:

```python
import json
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# Load config from checkpoint
lora_ckpt_dir = "/path/to/checkpoints/finetune_lora/step_0002000"
with open(f"{lora_ckpt_dir}/lora_config.json") as f:
    lora_info = json.load(f)

base_model = lora_info["base_model"]
lora_cfg = LoRAConfig(**lora_info["lora_config"])

# Load model with LoRA
model = VoxCPM.from_pretrained(
    hf_model_id=base_model,
    lora_config=lora_cfg,
    lora_weights_path=lora_ckpt_dir,
)
```

Or use the test script directly:

```bash
python scripts/test_voxcpm_lora_infer.py \
    --lora_ckpt /path/to/checkpoints/finetune_lora/step_0002000 \
    --text "Hello world"
```

### Method Reference

| Method | Description | torch.compile Compatible |
|--------|-------------|--------------------------|
| `load_lora(path)` | Load LoRA weights from file | ✅ |
| `set_lora_enabled(bool)` | Enable/disable LoRA | ✅ |
| `unload_lora()` | Reset LoRA weights to initial values | ✅ |
| `get_lora_state_dict()` | Get current LoRA weights | ✅ |
| `lora_enabled` | Property: check if LoRA is configured | ✅ |

---

## FAQ

### 1. How Much Data is Needed for LoRA Fine-tuning to Converge to a Single Voice?

We have tested with 5 minutes and 10 minutes of data (all audio clips are 3-6s in length). In our experiments, both datasets converged to a single voice after 2000 training steps with default configurations. You can adjust the data amount and training configurations based on your available data and computational resources.

### 2. Out of Memory (OOM)

- Increase `grad_accum_steps` (gradient accumulation)
- Decrease `batch_size`
- Use LoRA fine-tuning instead of full fine-tuning
- Decrease `max_batch_tokens` to filter long samples

### 3. Poor LoRA Performance

- Increase `r` (LoRA rank)
- Adjust `alpha` (try `alpha = r/2` or `alpha = r`)
- Increase training steps
- Add more target modules

### 4. Training Not Converging

- Decrease `learning_rate`
- Increase `warmup_steps`
- Check data quality

### 5. LoRA Not Taking Effect at Inference

- Check that `lora_config.json` exists in the checkpoint directory
- Check `load_lora()` return value - `skipped_keys` should be empty
- Verify `set_lora_enabled(True)` is called

### 6. Checkpoint Loading Errors

- Full fine-tuning: checkpoint directory should contain `model.safetensors` (or `pytorch_model.bin`), `config.json`, `audiovae.pth`
- LoRA: checkpoint directory should contain:
  - `lora_weights.safetensors` (or `lora_weights.ckpt`) - LoRA weights
  - `lora_config.json` - LoRA config and base model path
