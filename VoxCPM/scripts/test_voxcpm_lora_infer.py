#!/usr/bin/env python3
"""
LoRA inference test script.

Usage:

    python scripts/test_voxcpm_lora_infer.py \
        --lora_ckpt checkpoints/step_0002000 \
        --text "Hello, this is LoRA finetuned result." \
        --output lora_test.wav

With voice cloning:

    python scripts/test_voxcpm_lora_infer.py \
        --lora_ckpt checkpoints/step_0002000 \
        --text "This is voice cloning result." \
        --prompt_audio path/to/ref.wav \
        --prompt_text "Reference audio transcript" \
        --output lora_clone.wav

Note: The script reads base_model path and lora_config from lora_config.json
      in the checkpoint directory (saved automatically during training).
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig


def parse_args():
    parser = argparse.ArgumentParser("VoxCPM LoRA inference test")
    parser.add_argument(
        "--lora_ckpt",
        type=str,
        required=True,
        help="LoRA checkpoint directory (contains lora_weights.safetensors and lora_config.json)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="",
        help="Optional: override base model path (default: read from lora_config.json)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Target text to synthesize",
    )
    parser.add_argument(
        "--prompt_audio",
        type=str,
        default="",
        help="Optional: reference audio path for voice cloning",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="",
        help="Optional: transcript of reference audio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lora_test.wav",
        help="Output wav file path",
    )
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=2.0,
        help="CFG scale (default: 2.0)",
    )
    parser.add_argument(
        "--inference_timesteps",
        type=int,
        default=10,
        help="Diffusion inference steps (default: 10)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=600,
        help="Max generation steps",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable text normalization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Check LoRA checkpoint directory
    ckpt_dir = Path(args.lora_ckpt)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_dir}")

    # 2. Load lora_config.json from checkpoint
    lora_config_path = ckpt_dir / "lora_config.json"
    if not lora_config_path.exists():
        raise FileNotFoundError(
            f"lora_config.json not found in {ckpt_dir}. "
            "Make sure the checkpoint was saved with the updated training script."
        )
    
    with open(lora_config_path, "r", encoding="utf-8") as f:
        lora_info = json.load(f)
    
    # Get base model path (command line arg overrides config)
    pretrained_path = args.base_model if args.base_model else lora_info.get("base_model")
    if not pretrained_path:
        raise ValueError("base_model not found in lora_config.json and --base_model not provided")
    
    # Get LoRA config
    lora_cfg_dict = lora_info.get("lora_config", {})
    lora_cfg = LoRAConfig(**lora_cfg_dict) if lora_cfg_dict else None
    
    print(f"Loaded config from: {lora_config_path}")
    print(f"  Base model: {pretrained_path}")
    print(f"  LoRA config: r={lora_cfg.r}, alpha={lora_cfg.alpha}" if lora_cfg else "  LoRA config: None")

    # 3. Load model with LoRA (no denoiser)
    print(f"\n[1/2] Loading model with LoRA: {pretrained_path}")
    print(f"      LoRA weights: {ckpt_dir}")
    model = VoxCPM.from_pretrained(
        hf_model_id=pretrained_path,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=str(ckpt_dir),
    )

    # 4. Synthesize audio
    prompt_wav_path = args.prompt_audio if args.prompt_audio else None
    prompt_text = args.prompt_text if args.prompt_text else None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/2] Starting synthesis tests...")
    
    # === Test 1: With LoRA ===
    print(f"\n  [Test 1] Synthesize with LoRA...")
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    lora_output = out_path.with_stem(out_path.stem + "_with_lora")
    sf.write(str(lora_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {lora_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    # === Test 2: Disable LoRA (via set_lora_enabled) ===
    print(f"\n  [Test 2] Disable LoRA (set_lora_enabled=False)...")
    model.set_lora_enabled(False)
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    disabled_output = out_path.with_stem(out_path.stem + "_lora_disabled")
    sf.write(str(disabled_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {disabled_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    # === Test 3: Re-enable LoRA ===
    print(f"\n  [Test 3] Re-enable LoRA (set_lora_enabled=True)...")
    model.set_lora_enabled(True)
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    reenabled_output = out_path.with_stem(out_path.stem + "_lora_reenabled")
    sf.write(str(reenabled_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {reenabled_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    # === Test 4: Unload LoRA (reset_lora_weights) ===
    print(f"\n  [Test 4] Unload LoRA (unload_lora)...")
    model.unload_lora()
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    reset_output = out_path.with_stem(out_path.stem + "_lora_reset")
    sf.write(str(reset_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {reset_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    # === Test 5: Hot-reload LoRA (load_lora) ===
    print(f"\n  [Test 5] Hot-reload LoRA (load_lora)...")
    loaded, skipped = model.load_lora(ckpt_dir)
    print(f"           Reloaded {len(loaded)} parameters")
    audio_np = model.generate(
        text=args.text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
        normalize=args.normalize,
        denoise=False,
    )
    reload_output = out_path.with_stem(out_path.stem + "_lora_reloaded")
    sf.write(str(reload_output), audio_np, model.tts_model.sample_rate)
    print(f"           Saved: {reload_output}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")

    print(f"\n[Done] All tests completed!")
    print(f"  - with_lora:      {lora_output}")
    print(f"  - lora_disabled:  {disabled_output}")
    print(f"  - lora_reenabled: {reenabled_output}")
    print(f"  - lora_reset:     {reset_output}")
    print(f"  - lora_reloaded:  {reload_output}")


if __name__ == "__main__":
    main()
