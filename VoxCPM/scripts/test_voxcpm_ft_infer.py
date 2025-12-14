#!/usr/bin/env python3
"""
Full finetune inference script (no LoRA).

Checkpoint directory contains complete model files (pytorch_model.bin, config.json, audiovae.pth, etc.),
can be loaded directly via VoxCPM.

Usage:

    python scripts/test_voxcpm_ft_infer.py \
        --ckpt_dir /path/to/checkpoints/step_0001000 \
        --text "Hello, I am the finetuned VoxCPM." \
        --output ft_test.wav

With voice cloning:

    python scripts/test_voxcpm_ft_infer.py \
        --ckpt_dir /path/to/checkpoints/step_0001000 \
        --text "Hello, this is voice cloning result." \
        --prompt_audio path/to/ref.wav \
        --prompt_text "Reference audio transcript" \
        --output ft_clone.wav
"""

import argparse
from pathlib import Path

import soundfile as sf

from voxcpm.core import VoxCPM


def parse_args():
    parser = argparse.ArgumentParser("VoxCPM full-finetune inference test (no LoRA)")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory (contains pytorch_model.bin, config.json, audiovae.pth, etc.)",
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
        default="ft_test.wav",
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

    # Load model from checkpoint directory (no denoiser)
    print(f"[FT Inference] Loading model: {args.ckpt_dir}")
    model = VoxCPM.from_pretrained(
        hf_model_id=args.ckpt_dir,
        load_denoiser=False,
        optimize=True,
    )

    # Run inference
    prompt_wav_path = args.prompt_audio if args.prompt_audio else None
    prompt_text = args.prompt_text if args.prompt_text else None

    print(f"[FT Inference] Synthesizing: text='{args.text}'")
    if prompt_wav_path:
        print(f"[FT Inference] Using reference audio: {prompt_wav_path}")
        print(f"[FT Inference] Reference text: {prompt_text}")

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

    # Save audio
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio_np, model.tts_model.sample_rate)

    print(f"[FT Inference] Saved to: {out_path}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")


if __name__ == "__main__":
    main()
