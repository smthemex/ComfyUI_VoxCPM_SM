"""
VoxCPM: A Tokenizer-free speech generation model

This module contains the main VoxCPM model implementation, including configuration classes
and the core VoxCPMModel for text-to-speech generation.

Copyright 2026 OpenBMB
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from typing import Tuple, Union, Generator, List, Optional

import torch
import torch.nn as nn
import warnings
import librosa
import numpy as np
from einops import rearrange
from pydantic import BaseModel
import gc
try:
    from safetensors.torch import load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from ..modules.audiovae import AudioVAEV2, AudioVAEConfigV2
from ..modules.layers import ScalarQuantizationLayer
from ..modules.layers.lora import apply_lora_to_named_linear_modules
from ..modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiTV2
from ..modules.locenc import VoxCPMLocEnc
from ..modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import get_dtype, mask_multichar_chinese_tokens, next_and_close, resolve_runtime_device,apply_generation_seed,materialize_generation_seed
import math

# A simple function to trim audio silence using VAD, not used default
def _trim_audio_silence_vad(
        audio: torch.Tensor, sample_rate: int, max_silence_ms: float = 200.0, top_db: float = 35.0
    ) -> torch.Tensor:
    if audio.numel() == 0:
        return audio
    y = audio.squeeze(0).numpy()
    n = len(y)
    frame_length = 2048
    hop_length = 512
    ref = np.max(np.abs(y))
    if ref <= 0:
        return audio
    threshold = ref * (10.0 ** (-top_db / 20.0))

    try:
        _, (start, end) = librosa.effects.trim(
            y, top_db=top_db, ref=np.max, frame_length=frame_length, hop_length=hop_length
        )
    except Exception:
        start, end = 0, n

    # Find the last frame with continuous energy, trim the long pseudo-silence at the end (low energy background noise, etc.)
    n_frames = max(0, (n - frame_length) // hop_length + 1)
    last_voice_frame = -1
    for j in range(n_frames):
        idx = j * hop_length
        if idx + frame_length > n:
            break
        rms = np.sqrt(np.mean(y[idx : idx + frame_length] ** 2))
        if rms >= threshold:
            last_voice_frame = j
    if last_voice_frame >= 0:
        end_by_vad = min(n, (last_voice_frame + 1) * hop_length + (frame_length - hop_length))
        end = min(end, end_by_vad)

    max_silence_samples = int(max_silence_ms * sample_rate / 1000.0)
    new_start = max(0, start - max_silence_samples)
    new_end = min(n, end + max_silence_samples)
    return audio[:, new_start:new_end]


class VoxCPMEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None


class VoxCPMDitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None
    dit_mean_mode: bool = False

    cfm_config: CfmConfig


class VoxCPMConfig(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 4
    feat_dim: int = 64
    residual_lm_num_layers: int = 8
    residual_lm_no_rope: bool = False
    scalar_quantization_latent_dim: int = 512
    scalar_quantization_scale: int = 9

    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig
    audio_vae_config: Optional[AudioVAEConfigV2] = None

    max_length: int = 8192
    device: str = "cuda"
    dtype: str = "bfloat16"


class LoRAConfig(BaseModel):
    enable_lm: bool = False  # Apply LoRA to base_lm + residual_lm
    enable_dit: bool = False  # Apply LoRA to VoxCPMLocDiT
    enable_proj: bool = False  # Apply LoRA to projection Linear layers

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0

    # Target linear layer names for LM & DiT (matched by attribute name)
    target_modules_lm: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    target_modules_dit: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    # Projection layer attribute names to find on VoxCPM2Model
    target_proj_modules: list[str] = ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"]


VoxCPMConfig.model_rebuild()


class VoxCPM2Model(nn.Module):
    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAEV2,
        lora_config: LoRAConfig = None,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.device = resolve_runtime_device(device, config.device)
        self.config.device = self.device
        print(f"Running on device: {self.device}, dtype: {self.config.dtype}", file=sys.stderr)

        # Text-Semantic LM
        self.base_lm = MiniCPMModel(config.lm_config)
        self.base_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104
        self.last_successful_seed = None

        # Residual Acoustic LM
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        residual_lm_config.no_rope = config.residual_lm_no_rope
        self.residual_lm = MiniCPMModel(residual_lm_config)
        self.residual_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiTV2(decoder_config, in_channels=config.feat_dim),
            mean_mode=config.dit_config.dit_mean_mode,
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.fusion_concat_proj = nn.Linear(config.lm_config.hidden_size * 2, config.lm_config.hidden_size)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
        self.stop_loss = nn.CrossEntropyLoss(reduction="none")

        # Audio VAE
        use_gguf = kwargs.get("use_gguf", False) 
        self.audio_vae = audio_vae
        if use_gguf :
            self.chunk_size=math.prod([2, 5, 8, 8])
            self.decode_chunk_size = math.prod([8, 6, 5, 2, 2, 2])
            self._decode_chunk_size = getattr(audio_vae, "decode_chunk_size", self.decode_chunk_size)
            self._encode_sample_rate = 16000
            self.sample_rate = 48000
        else:
            self.chunk_size = audio_vae.chunk_size
            self._decode_chunk_size = getattr(audio_vae, "decode_chunk_size", audio_vae.chunk_size)  
            self._encode_sample_rate = audio_vae.sample_rate
            self.sample_rate = getattr(audio_vae, "out_sample_rate", audio_vae.sample_rate)
        



    def _apply_lora(self):
        """注入 LoRA 到 LM / DiT / 投影层"""
        cfg = self.lora_config
        lora_kwargs = dict(r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)

        # LM: base_lm + residual_lm
        if cfg.enable_lm:
            for lm in [self.base_lm, self.residual_lm]:
                apply_lora_to_named_linear_modules(lm, target_submodule_names=cfg.target_modules_lm, **lora_kwargs)

        # DiT: feat_decoder.estimator
        if cfg.enable_dit:
            apply_lora_to_named_linear_modules(
                self.feat_decoder.estimator, target_submodule_names=cfg.target_modules_dit, **lora_kwargs
            )

        # 投影层
        if cfg.enable_proj:
            from ..modules.layers.lora import LoRALinear

            for attr_name in cfg.target_proj_modules:
                module = getattr(self, attr_name, None)
                if isinstance(module, nn.Linear):
                    setattr(self, attr_name, LoRALinear(base=module, **lora_kwargs))

    def optimize(self, disable: bool = False):
        if disable:
            return self
        try:
            if self.device != "cuda":
                raise ValueError("VoxCPMModel can only be optimized on CUDA device")
            try:
                import triton  # noqa: F401
            except ImportError:
                raise ValueError("triton is not installed")
            # FP8 / W4A8 量化层包含 float8_e4m3fn 张量，与 torch.compile 的
            # reduce-overhead（CUDA graph）模式不兼容：图捕获时 fp8 权重的
            # 反量化 cast 会被静默错误处理，得到全零/垃圾权重 -> 推理声音空白。
            # 此类模型跳过 CUDA-graph 编译，退回普通前向以保证正确性。
            has_fp8_like = any(
                isinstance(m, (ConvRotFp8Linear, ConvRotW4A8Linear))
                for m in self.modules()
            )
            if has_fp8_like:
                print("[ConvRot] 检测到 FP8/W4A8 量化层，已禁用 CUDA-graph（torch.compile reduce-overhead）"
                      "编译，以避免 fp8 张量在编译图中被错误处理导致推理空白。退回普通前向（保证正确，速度略降）。",
                      file=sys.stderr)
                return self
            self.base_lm.forward_step = torch.compile(self.base_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.residual_lm.forward_step = torch.compile(
                self.residual_lm.forward_step, mode="reduce-overhead", fullgraph=True
            )
            self._feat_encoder_raw = self.feat_encoder
            self.feat_encoder = torch.compile(self.feat_encoder, mode="reduce-overhead", fullgraph=True)
            self.feat_decoder.estimator = torch.compile(
                self.feat_decoder.estimator, mode="reduce-overhead", fullgraph=True
            )
        except Exception as e:
            print(f"Warning: torch.compile disabled - {e}", file=sys.stderr)
        return self

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        progress: float = 0.0,
        sample_generate: bool = False,
    ):
        del position_ids  # not used yet

        text_tokens = text_tokens.to(self.device, dtype=torch.long)
        text_mask = text_mask.to(self.device, dtype=self._dtype())
        audio_feats = audio_feats.to(self.device, dtype=self._dtype())
        audio_mask = audio_mask.to(self.device, dtype=self._dtype())
        loss_mask = loss_mask.to(self.device, dtype=self._dtype())
        labels = labels.to(self.device, dtype=torch.long)

        B, T, P, D = audio_feats.shape
        feat_embed = self.feat_encoder(audio_feats)
        feat_embed = self.enc_to_lm_proj(feat_embed)

        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
        text_embed = self.base_lm.embed_tokens(text_tokens) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

        enc_outputs, _ = self.base_lm(inputs_embeds=combined_embed, is_causal=True)
        enc_outputs = enc_outputs.to(self._dtype())
        enc_outputs = self.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = torch.cat((torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)

        residual_inputs = self.fusion_concat_proj(
            torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_outputs, _ = self.residual_lm(inputs_embeds=residual_inputs, is_causal=True)
        residual_outputs = residual_outputs.to(self._dtype())
        residual_hidden = torch.cat(
            (torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]),
            dim=1,
        )

        dit_hidden = torch.cat((self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(residual_hidden)), dim=-1)
        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")

        # Keep diffusion inputs in the same dtype as the model (e.g., bfloat16)
        target_dtype = self._dtype()

        feat_gt = rearrange(audio_feats.to(target_dtype), "b t p d -> (b t) p d")
        feat_cond = torch.cat(
            (torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]),
            dim=1,
        )
        feat_cond = rearrange(feat_cond.to(target_dtype), "b t p d -> (b t) p d")

        loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        loss_seq_mask = rearrange(loss_seq_mask, "b t p -> (b t) p 1").to(target_dtype)

        diff_loss = self.feat_decoder.compute_loss(
            feat_gt.transpose(1, 2).contiguous(),
            dit_hidden,
            cond=feat_cond.transpose(1, 2).contiguous(),
            tgt_mask=loss_seq_mask.transpose(1, 2).contiguous(),
            progress=progress,
        )

        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
        stop_losses = self.stop_loss(stop_logits.transpose(1, 2), labels)
        denom = torch.clamp(loss_mask.sum(), min=1.0)
        stop_loss = (stop_losses * loss_mask).sum() / denom

        feat_pred = None
        if sample_generate:
            feat_cond_for_sample = feat_cond.transpose(1, 2).contiguous()
            feat_pred_seq = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=feat_cond_for_sample,
                n_timesteps=10,
            )
            feat_pred = rearrange(feat_pred_seq.transpose(1, 2), "(b t) d p -> b d (t p)", b=B, p=self.patch_size)

        feat_gt_tensor = rearrange(feat_gt, "(b t) p d -> b d (t p)", b=B, p=self.patch_size)

        return {
            "loss/diff": diff_loss,
            "loss/stop": stop_loss,
            "feat_gt": feat_gt_tensor,
            "feat_pred": feat_pred,
        }

    def _dtype(self):
        return get_dtype(self.config.dtype)

    def _encode_wav(
        self,
        wav_path: str,
        padding_mode: str = "right",
        trim_silence_vad: bool = False,
    ) -> torch.Tensor:
        """Load, trim, pad and VAE-encode an audio file.

        Args:
            wav_path: path to the audio file.
            padding_mode: "right" (default) or "left" padding for alignment.
            trim_silence_vad: whether to apply VAD-based silence trimming.

        Returns:
            audio_feat: (T, P, D) tensor of latent patches.
        """
        audio, _ = librosa.load(wav_path, sr=self._encode_sample_rate, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)
        if trim_silence_vad:
            audio = _trim_audio_silence_vad(audio, self._encode_sample_rate, max_silence_ms=200.0)
        patch_len = self.patch_size * self.chunk_size
        if audio.size(1) % patch_len != 0:
            padding_size = patch_len - audio.size(1) % patch_len
            pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
            audio = torch.nn.functional.pad(audio, pad)
        feat = self.audio_vae.encode(audio.to(self.device), self._encode_sample_rate).cpu()
        return feat.view(self.audio_vae.latent_dim, -1, self.patch_size).permute(1, 2, 0)

    def _make_ref_prefix(self, ref_feat: torch.Tensor, device: torch.device):
        """Build the [ref_start ref_audio ref_end] prefix segments.

        Returns:
            tokens, feats, text_mask, audio_mask  (all 1-D / 2-D tensors)
        """
        ref_len = ref_feat.size(0)
        z1 = torch.zeros((1, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32, device=device)
        tokens = torch.cat(
            [
                torch.tensor([self.ref_audio_start_token], dtype=torch.int32, device=device),
                torch.zeros(ref_len, dtype=torch.int32, device=device),
                torch.tensor([self.ref_audio_end_token], dtype=torch.int32, device=device),
            ]
        )
        feats = torch.cat([z1, ref_feat, z1], dim=0)
        t_mask = torch.cat(
            [
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(ref_len, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
            ]
        ).to(device)
        a_mask = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.ones(ref_len, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            ]
        ).to(device)
        return tokens, feats, t_mask, a_mask

    def generate(self, *args, **kwargs) -> torch.Tensor:
        return next_and_close(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        reference_wav_path: str = "",
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        trim_silence_vad: bool = False,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
        seed: Optional[int] = None,
    ) -> Generator[torch.Tensor, None, None]:
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False

        if reference_wav_path and prompt_wav_path:
            # Combined mode: reference isolation prefix + continuation suffix
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(
                reference_wav_path,
                padding_mode="right",
                trim_silence_vad=trim_silence_vad,
            )
            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left", trim_silence_vad=trim_silence_vad)
            prompt_audio_length = prompt_feat.size(0)

            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, text_token.device)

            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )

            text_token = torch.cat([ref_tokens, text_token, prompt_pad_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat, prompt_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                    torch.zeros(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                    torch.ones(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        elif reference_wav_path:
            # Reference-only mode (prompt isolation)
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(
                reference_wav_path,
                padding_mode="right",
                trim_silence_vad=trim_silence_vad,
            )
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, text_token.device)

            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([ref_tokens, text_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        elif len(prompt_wav_path) == 0:
            # Zero-shot mode
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_mask = torch.ones(text_length, dtype=torch.int32).to(text_token.device)
            audio_mask = torch.zeros(text_length, dtype=torch.int32).to(text_token.device)

        else:
            # Continuation-only mode
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left", trim_silence_vad=trim_silence_vad)
            prompt_audio_length = prompt_feat.size(0)
            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, prompt_pad_token])
            audio_feat = torch.cat([text_pad_feat, prompt_feat], dim=0)
            text_mask = torch.cat(
                [
                    torch.ones(text_length, dtype=torch.int32),
                    torch.zeros(prompt_audio_length, dtype=torch.int32),
                ]
            ).to(text_token.device)
            audio_mask = torch.cat(
                [
                    torch.zeros(text_length, dtype=torch.int32),
                    torch.ones(prompt_audio_length, dtype=torch.int32),
                ]
            ).to(text_token.device)

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        target_text_length = len(self.text_tokenizer(target_text))

        retry_badcase_times = 0
        current_seed = materialize_generation_seed(seed)
        last_attempt_seed = current_seed
        while retry_badcase_times < retry_badcase_max_times:
            last_attempt_seed = current_seed
            apply_generation_seed(last_attempt_seed)
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                with self.audio_vae.streaming_decode() as vae_dec:
                    for latent_pred, _, _ctx in inference_result:
                        decode_audio = vae_dec.decode_chunk(latent_pred.to(torch.float32))
                        decode_audio = decode_audio.squeeze(1).cpu()
                        self.last_successful_seed = last_attempt_seed
                        yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat, context_len = next_and_close(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(
                            f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                            file=sys.stderr,
                        )
                        retry_badcase_times += 1
                        current_seed += 1
                        continue
                    else:
                        break
                else:
                    break

        if not streaming:
            self.last_successful_seed = last_attempt_seed
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            decode_patch_len = self.patch_size * self._decode_chunk_size
            if context_len > 0:
                decode_audio = decode_audio[..., decode_patch_len * context_len:].squeeze(1).cpu()
            else:
                decode_audio = decode_audio.squeeze(1).cpu()
            yield decode_audio

    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str = None,
        prompt_wav_path: str = None,
        reference_wav_path: str = None,
        trim_silence_vad: bool = False,
    ):
        """
        Build prompt cache for subsequent generation.

        Supports the same parameter combinations as ``generate()``:
        - ``reference_wav_path`` only -> reference mode (voice cloning, isolated)
        - ``prompt_text`` + ``prompt_wav_path`` -> continuation mode
        - all three -> combined ref + continuation mode

        Args:
            prompt_text: prompt text for continuation mode.
                Must be paired with ``prompt_wav_path``.
            prompt_wav_path: prompt audio path for continuation mode.
                Must be paired with ``prompt_text``.
            reference_wav_path: reference audio path for voice cloning
                (structurally isolated via ref_audio tokens).
            trim_silence_vad: whether to apply VAD-based silence trimming
                before encoding prompt/reference audio.

        Returns:
            prompt_cache: dict used by ``_generate_with_prompt_cache``.
        """
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        if prompt_wav_path is None and reference_wav_path is None:
            raise ValueError("At least one of prompt_wav_path or reference_wav_path must be provided")

        cache = {}

        if reference_wav_path:
            cache["ref_audio_feat"] = self._encode_wav(
                reference_wav_path,
                padding_mode="right",
                trim_silence_vad=trim_silence_vad,
            )

        if prompt_wav_path and prompt_text is not None:
            cache["prompt_text"] = prompt_text
            cache["audio_feat"] = self._encode_wav(
                prompt_wav_path,
                padding_mode="left",
                trim_silence_vad=trim_silence_vad,
            )

        has_ref = "ref_audio_feat" in cache
        has_prompt = "audio_feat" in cache
        if has_ref and has_prompt:
            cache["mode"] = "ref_continuation"
        elif has_ref:
            cache["mode"] = "reference"
        else:
            cache["mode"] = "continuation"

        return cache

    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text: str,
        new_audio_feat: torch.Tensor,
    ):
        """
        Merge original prompt cache with newly generated content to stabilize voice.

        Args:
            original_cache: original prompt cache (any mode)
            new_text: newly generated text
            new_audio_feat: newly generated audio features

        Returns:
            merged_cache: merged cache with prompt_text and audio_feat
        """
        if original_cache is None:
            return {
                "prompt_text": new_text,
                "audio_feat": new_audio_feat,
                "mode": "continuation",
            }
        merged = {}
        if "ref_audio_feat" in original_cache:
            merged["ref_audio_feat"] = original_cache["ref_audio_feat"]
        merged["prompt_text"] = original_cache.get("prompt_text", "") + new_text
        old_feat = original_cache.get("audio_feat", new_audio_feat.new_empty(0, *new_audio_feat.shape[1:]))
        merged["audio_feat"] = torch.cat([old_feat, new_audio_feat], dim=0)
        merged["mode"] = "ref_continuation" if "ref_audio_feat" in merged else "continuation"
        return merged

    def generate_with_prompt_cache(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next_and_close(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))

    def generate_with_prompt_cache_streaming(
        self, *args, **kwargs
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], None, None]:
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
        seed: Optional[int] = None,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Generate audio using pre-built prompt cache.

        Args:
            target_text: Text to convert to speech
            prompt_cache: Cache built by ``build_prompt_cache()``. Can be None
                for zero-shot generation.
            min_len: Minimum audio length to avoid very short audio
            max_len: Maximum audio length
            inference_timesteps: Number of diffusion sampling steps
            cfg_value: Classifier-free guidance value
            retry_badcase: Whether to retry on bad cases
            retry_badcase_max_times: Maximum retry attempts
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio
            streaming: Whether to return a generator of audio chunks
            streaming_prefix_len: Number of prefix audio patches to use for streaming mode

        Returns:
            Generator of Tuple containing:
                - Decoded audio tensor for the current step if ``streaming=True``, else final decoded audio tensor
                - Tensor of new text tokens
                - New audio features up to the current step as a List if ``streaming=True``, else as a concatenated Tensor
        """
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False

        # Determine mode from cache
        if prompt_cache is None:
            mode = "zero_shot"
            text = target_text
        else:
            mode = prompt_cache.get("mode", "continuation")
            if mode in ("continuation", "ref_continuation"):
                prompt_text = prompt_cache.get("prompt_text", "")
                text = prompt_text + target_text
            else:
                text = target_text

        text_token = torch.LongTensor(self.text_tokenizer(text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
            ],
            dim=-1,
        )

        target_text_token = torch.LongTensor(self.text_tokenizer(target_text))
        text_length = text_token.shape[0]

        if mode in ("zero_shot", "continuation"):
            prompt_audio_feat = (
                prompt_cache["audio_feat"]
                if prompt_cache
                else torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
            )
            audio_length = prompt_audio_feat.size(0)
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, text_pad_token])
            audio_feat = torch.cat([text_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat(
                [torch.ones(text_length, dtype=torch.int32), torch.zeros(audio_length, dtype=torch.int32)]
            ).to(text_token.device)
            audio_mask = torch.cat(
                [torch.zeros(text_length, dtype=torch.int32), torch.ones(audio_length, dtype=torch.int32)]
            ).to(text_token.device)

        elif mode == "reference":
            ref_audio_feat = prompt_cache["ref_audio_feat"]
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_audio_feat, text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([ref_tokens, text_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
            text_mask = torch.cat([ref_t_mask, torch.ones(text_length, dtype=torch.int32).to(text_token.device)])
            audio_mask = torch.cat([ref_a_mask, torch.zeros(text_length, dtype=torch.int32).to(text_token.device)])

        else:
            # ref_continuation mode
            ref_audio_feat = prompt_cache["ref_audio_feat"]
            prompt_audio_feat = prompt_cache["audio_feat"]
            prompt_audio_length = prompt_audio_feat.size(0)

            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_audio_feat, text_token.device)

            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )

            text_token = torch.cat([ref_tokens, text_token, prompt_pad_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                    torch.zeros(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                    torch.ones(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        # run inference
        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        current_seed = materialize_generation_seed(seed)
        last_attempt_seed = current_seed
        while retry_badcase_times < retry_badcase_max_times:
            last_attempt_seed = current_seed
            apply_generation_seed(last_attempt_seed)
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                with self.audio_vae.streaming_decode() as vae_dec:
                    for latent_pred, pred_audio_feat, _ctx in inference_result:
                        decode_audio = vae_dec.decode_chunk(latent_pred.to(torch.float32))
                        decode_audio = decode_audio.squeeze(1).cpu()
                        self.last_successful_seed = last_attempt_seed
                        yield (decode_audio, target_text_token, pred_audio_feat)
                break
            else:
                latent_pred, pred_audio_feat, context_len = next_and_close(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(
                            f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                            file=sys.stderr,
                        )
                        retry_badcase_times += 1
                        current_seed += 1
                        continue
                    else:
                        break
                else:
                    break
        if not streaming:
            self.last_successful_seed = last_attempt_seed
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            decode_patch_len = self.patch_size * self._decode_chunk_size
            if context_len > 0:
                decode_audio = decode_audio[..., decode_patch_len * context_len:].squeeze(1).cpu()
            else:
                decode_audio = decode_audio.squeeze(1).cpu()
            yield (decode_audio, target_text_token, pred_audio_feat)

    def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_pred, generated_feat, _ = next_and_close(self._inference(*args, streaming=False, **kwargs))
        return feat_pred, generated_feat

    def inference_streaming(self, *args, **kwargs) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
        for feat_pred, pred_feat_seq, _ in self._inference(*args, streaming=True, **kwargs):
            yield feat_pred, pred_feat_seq

    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """Core inference method for audio generation.

        This is the main inference loop that generates audio features
        using the language model and diffusion transformer.

        Args:
            text: Input text tokens
            text_mask: Mask for text tokens
            feat: Input audio features
            feat_mask: Mask for audio features
            min_len: Minimum generation length
            max_len: Maximum generation length
            inference_timesteps: Number of diffusion steps
            cfg_value: Classifier-free guidance value
            streaming: Whether to yield each step latent feature or just the final result

        Returns:
            Generator of Tuple containing:
                - Predicted latent feature at the current step if ``streaming=True``, else final latent features
                - Predicted audio feature sequence so far as a List if ``streaming=True``, else as a concatenated Tensor
        """
        B, T, P, D = feat.shape

        prefill_encoder = getattr(self, "_feat_encoder_raw", self.feat_encoder)
        feat_embed = prefill_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)

        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0

        text_embed = self.base_lm.embed_tokens(text) * scale_emb
       
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed #torch.Size([1, 76, 2048]) torch.Size([1, 76, 2048]) torch.Size([1, 76]) torch.Size([1, 76])

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        pred_feat_seq = []  # b, t, p, d
        curr_embed = None

        # Prepare prompt context patches for streaming mode
        # - Continuation modes (feat_mask ends with 1): use the last (streaming_prefix_len - 1)
        #   trailing audio patches as initial context so the VAE can decode smoothly.
        # - Reference-only / zero-shot (feat_mask ends with 0): start from scratch.
        has_continuation_audio = feat_mask[0, -1].item() == 1
        context_len = 0
        if has_continuation_audio:
            audio_indices = feat_mask.squeeze(0).nonzero(as_tuple=True)[0]
            context_len = min(streaming_prefix_len - 1, len(audio_indices))
            last_audio_indices = audio_indices[-context_len:]
            pred_feat_seq = list(feat[:, last_audio_indices, :, :].split(1, dim=1))
        else:
            pred_feat_seq = []

        enc_outputs, kv_cache_tuple = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        residual_enc_inputs = self.fusion_concat_proj(
            torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=residual_enc_inputs,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]

        for i in tqdm(range(max_len)):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(
                1, 2
            )  # [b, p, d]

            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)

            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat

            if streaming:
                # Yield only the newest patch latent for stateful VAE decode
                feat_pred = rearrange(pred_feat.unsqueeze(1), "b t p d -> b d (t p)", b=B, p=self.patch_size)

                yield feat_pred, pred_feat_seq, context_len

                if len(pred_feat_seq) > streaming_prefix_len:
                    pred_feat_seq = pred_feat_seq[-streaming_prefix_len:]

            stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
            if i > min_len and stop_flag == 1:
                break

            lm_hidden = self.base_lm.forward_step(
                curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()

            lm_hidden = self.fsq_layer(lm_hidden)
            curr_residual_input = self.fusion_concat_proj(torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1))
            residual_hidden = self.residual_lm.forward_step(
                curr_residual_input, torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()

        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
            generated_feat = pred_feat_seq[:, context_len:, :, :].squeeze(0).cpu()
            yield feat_pred, generated_feat, context_len

    @classmethod
    def from_local(
        cls,
        path: str,
        optimize: bool = True,
        training: bool = False,
        device: str | None = None,
        lora_config: LoRAConfig = None,
        **kwargs,
    ):
        vae_path = kwargs.get("vae_path", None)
        if not (not training and kwargs.get("gguf_path", None) is not None):
            assert os.path.exists(vae_path), f"VAE weights not found at {vae_path}"
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as _cfg_f:
            config = VoxCPMConfig.model_validate_json(_cfg_f.read())
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        audio_vae_config = getattr(config, "audio_vae_config", None)
        audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()

        if kwargs.get("gguf_path") is not None and not training:
            device=resolve_runtime_device(device)   
            model = cls(config, tokenizer, None, None, device=device, use_gguf=True)
            model_state_dict=load_gguf_checkpoint(kwargs.get("gguf_path"))
            set_gguf2meta_model(model, model_state_dict, torch.bfloat16, "cpu")
            if lora_config is not None:
                model.lora_config = lora_config
                model._apply_lora()
            if vae_path.endswith(".safetensors"):
                vae_state_dict = load_file(vae_path, device="cpu")["state_dict"]
            else:
                vae_state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)["state_dict"]
            x = audio_vae.load_state_dict(vae_state_dict, strict=False)
            print(x)
            model.audio_vae = audio_vae.to(torch.float32)
            return model.to(device).eval().optimize(disable=not optimize)

        # --- normal (non-gguf) path ---
        model = cls(config, tokenizer, audio_vae, lora_config, device=device)

        # Load state dicts
        if vae_path.endswith(".safetensors"):
            vae_state_dict = load_file(vae_path, device="cpu")["state_dict"]
        else:
            vae_state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)["state_dict"]
        ckpt_path = kwargs.get("ckpt_path", None)
        assert ckpt_path is not None, "Please provide 'ckpt_path' in kwargs for model weights."
        if ckpt_path.endswith('.safetensors'):
            model_state_dict = load_file(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model_state_dict = checkpoint.get("state_dict", checkpoint)
        for kw, val in vae_state_dict.items():
            model_state_dict[f"audio_vae.{kw}"] = val

        # ========== ConvRot INT8 / INT4 / FP8 / W4A8 detection & handling ==========
        # convrot_plan: base_path -> format
        #   "int8_tensorwise"  -> ConvRotLinear（INT8 ConvRot）
        #   "convrot_w4a4"     -> ConvRotInt4Linear（INT4 ConvRot）
        #   "scaled_fp8"       -> ConvRotFp8Linear（FP8 scaled，无 ConvRot 旋转）
        #   "asym_w4a8_int8"   -> ConvRotW4A8Linear（W4A8 ConvRot）
        convrot_plan = {}
        convrot_meta = {}
        for k in list(model_state_dict.keys()):
            if k.endswith(".comfy_quant"):
                base = k[:-len(".comfy_quant")]
                try:
                    conf = json.loads(model_state_dict[k].numpy().tobytes())
                    fmt = conf.get("format")
                    # 只要存在 comfy_quant 元数据且格式受支持，即视为量化层。
                    # 注意：quant_w4a8_convrot.py 导出的元数据只有
                    # {format, group_size, convrot_groupsize}，不含 "convrot" 字段，
                    # 因此不能要求 conf.get("convrot") 为真。
                    if fmt in _CONVROT_CLASS_BY_FMT:
                        convrot_plan[base] = fmt
                        convrot_meta[base] = conf
                except Exception:
                    pass
            elif k.endswith(".scaled_fp8"):
                # 兼容 ComfyUI 原生 scaled_fp8 布局：scope 级标量 marker（scope + "scaled_fp8"）
                # 该 marker 存在时，把所有 2D fp8 权重层都视为 scaled_fp8
                if "__fp8_marker_seen__" not in convrot_meta:
                    convrot_meta["__fp8_marker_seen__"] = True
                    scope = k[:-len("scaled_fp8")]  # 可能是空串或某前缀
                    for wk in list(model_state_dict.keys()):
                        if (wk.endswith(".weight")
                                and model_state_dict[wk].dtype == torch.float8_e4m3fn
                                and wk.startswith(scope)):
                            base = wk[:-len(".weight")]
                            if base not in convrot_plan:
                                convrot_plan[base] = "scaled_fp8"
                                convrot_meta[base] = {"format": "scaled_fp8", "convrot": True}
            elif k.endswith(".scale_inverted"):
                # 本项目专用 fp8_voxcpm 格式：逐层逆 scale（= 1/scale）
                base = k[:-len(".scale_inverted")]
                if base not in convrot_plan:
                    convrot_plan[base] = "fp8_voxcpm"
                    convrot_meta[base] = {"format": "fp8_voxcpm", "convrot": True}
            elif k.endswith(".scale_weight"):
                # 原生 scaled_fp8 格式：逐层正 scale（= 除数 scale）
                base = k[:-len(".scale_weight")]
                if base not in convrot_plan:
                    convrot_plan[base] = "scaled_fp8"
                    convrot_meta[base] = {"format": "scaled_fp8", "convrot": True}

        if convrot_plan:
            n8 = sum(1 for f in convrot_plan.values() if f == "int8_tensorwise")
            n4 = sum(1 for f in convrot_plan.values() if f == "convrot_w4a4")
            nfp8 = sum(1 for f in convrot_plan.values() if f == "scaled_fp8")
            nw4a8 = sum(1 for f in convrot_plan.values() if f == "asym_w4a8_int8")
            print(f"[ConvRot] Detected {len(convrot_plan)} convrot layers "
                  f"(int8={n8}, int4={n4}, fp8={nfp8}, w4a8={nw4a8}), "
                  f"replacing specific Linear modules...", file=sys.stderr)
            # 删除 comfy_quant 标记（不再需要）
            for k in list(model_state_dict.keys()):
                if k.endswith(".comfy_quant"):
                    base = k[:-len(".comfy_quant")]
                    if base in convrot_plan:
                        del model_state_dict[k]
            # 维度预检：checkpoint 量化权重的 in/out 维度必须与模型（config）一致。
            # 量化不改变维度，若不一致说明该 checkpoint 来自不同规模（不同 hidden_size）的模型。
            for base, fmt in convrot_plan.items():
                w_key = f"{base}.weight"
                if w_key not in model_state_dict:
                    continue
                ckpt_w = model_state_dict[w_key]
                mod = _find_module_by_key(model, base)
                if mod is None or not hasattr(mod, "out_features"):
                    continue
                # convrot 量化权重第一维=out_features；packed 类第二维为 in//2
                if fmt in ("convrot_w4a4", "asym_w4a8_int8"):
                    exp_out = mod.out_features
                    exp_in = mod.in_features // 2
                else:
                    exp_out, exp_in = mod.out_features, mod.in_features
                if ckpt_w.shape[0] != exp_out or ckpt_w.shape[1] != exp_in:
                    raise RuntimeError(
                        f"[ConvRot] 维度不匹配：量化层 '{base}' 的 checkpoint 权重形状为 "
                        f"{tuple(ckpt_w.shape)}，但模型（config）期望 [out={exp_out}, in={exp_in}]。"
                        f"这通常意味着该量化 checkpoint 来自不同规模（不同 hidden_size）的模型，"
                        f"请确认加载的 config.json 与导出该量化文件所用的原始模型一致。"
                    )
            # 按格式替换 convrot 量化层
            # 必须发生在 .to() 之前
            model, replaced_keys = replace_linear_with_convrot(model, convrot_plan)
            # 将检测到的格式写入各 ConvRot 模块，供反量化时精确选择 scale
            for base, fmt in convrot_plan.items():
                mod = _find_module_by_key(model, base)
                if mod is not None and isinstance(mod, _CONVROT_LAYER_TYPES):
                    mod._convrot_format = fmt
            # 针对 W4A8 层：根据元数据调整 group_size / convrot_groupsize 并 resize s_rel buffer，
            # 使其与磁盘上的 [out, in//group_size] 形状一致（load_state_dict 才能复制）。
            for base, fmt in convrot_plan.items():
                if fmt != "asym_w4a8_int8":
                    continue
                mod = _find_module_by_key(model, base)
                if mod is None or not isinstance(mod, ConvRotW4A8Linear):
                    continue
                conf = convrot_meta.get(base, {})
                gs = int(conf.get("group_size", 16))
                cgs = int(conf.get("convrot_groupsize", 256))
                mod.group_size = gs
                mod.convrot_groupsize = cgs
                # 重新注册 s_rel buffer 为正确形状（dtype 用 fp32，避免 fp8 下溢归零）
                dev = mod.weight_s_rel.device
                mod.register_buffer(
                    'weight_s_rel',
                    torch.empty((mod.out_features, mod.in_features // gs),
                                dtype=torch.float32, device=dev),
                )
            # 修改 state dict key：被量化层的 .weight -> 对应 buffer
            #   int8  -> .weight_int8
            #   int4  -> .weight_packed
            #   fp8   -> .weight_fp8
            #   w4a8  -> .weight_packed（int4 打包）+ 各 scale buffer
            # 其余量化专属 buffer（scale_inverted / s_rel / s_channel / codebook）直接按原 key 保留
            for base, fmt in convrot_plan.items():
                w_key = f"{base}.weight"
                if w_key in model_state_dict:
                    buf_key = {
                        "int8_tensorwise": "weight_int8",
                        "convrot_w4a4": "weight_packed",
                        "scaled_fp8": "weight_fp8",
                        "asym_w4a8_int8": "weight_packed",
                    }.get(fmt, "weight_int8")
                    model_state_dict[f"{base}.{buf_key}"] = model_state_dict.pop(w_key)
            # W4A8 兼容：buffer 统一 fp32（避免 fp8 下溢）。若 checkpoint 的 weight_s_rel
            # 以 fp8 导出（--scale_dtype fp8），则就地 .float() 转为 fp32 以对齐 buffer dtype。
            for base, fmt in convrot_plan.items():
                if fmt != "asym_w4a8_int8":
                    continue
                rel_key = f"{base}.weight_s_rel"
                if rel_key in model_state_dict and model_state_dict[rel_key].dtype != torch.float32:
                    model_state_dict[rel_key] = model_state_dict[rel_key].to(torch.float32)
            print(f"[ConvRot] State dict keys adjusted; loading convrot weights into layers.", file=sys.stderr)
        else:
            print("[ConvRot] No convrot layers detected.", file=sys.stderr)

        # ========== LoRA 注入：必须放在 convrot 替换之后 ==========
        # 此时 convrot 层已被替换为 ConvRotLinear（不是 nn.Linear），
        # _apply_lora() 的 apply_lora_to_named_linear_modules 会自然跳过它们，
        # 只替换真正的 nn.Linear 为 LoRALinear。
        if lora_config is not None:
            model._apply_lora()
            print(f"[LoRA] Applied LoRA to remaining nn.Linear layers (auto-skipped ConvRotLinear).", file=sys.stderr)

        # Convert model to compute dtype *after* convrot replacement (ConvRotLinear._apply protects int8/scale buffers)
        if not training:
            lm_dtype = get_dtype(model.config.dtype)
            model = model.to(lm_dtype)
        else:
            for name, param in model.named_parameters():
                if "audio_vae" in name:
                    param.requires_grad = False
                    continue
                if lora_config is not None and "lora" not in name:
                    param.requires_grad = False
        model.audio_vae = model.audio_vae.to(torch.float32)

        # embed_tokens 可能被量化脚本量化（但它是 nn.Embedding，不会被
        # replace_linear_with_convrot 替换），这里手动反量化成普通 Embedding.weight，
        # 否则 embed 随机初始化 -> 灾难性错误（推理空白/乱码）。
        # 支持两种量化风格的 embed：
        #   - FP8 (scaled_fp8 / fp8_voxcpm)：*.weight_fp8 + *.scale_weight(或 *.scale_inverted)
        #   - W4A8 (asym_w4a8_int8)：*.weight_packed + *.weight_s_rel + *.weight_s_channel + *.weight_codebook
        emb_prefix = "base_lm.embed_tokens"
        emb = model.base_lm.embed_tokens
        try:
            if f"{emb_prefix}.weight_fp8" in model_state_dict:
                # ---- FP8 风格 embed ----
                wq = model_state_dict.pop(f"{emb_prefix}.weight_fp8").float()
                sc = model_state_dict.pop(f"{emb_prefix}.scale_weight", None)
                if sc is None:
                    sc = model_state_dict.pop(f"{emb_prefix}.scale_inverted", None)
                if sc is not None:
                    if sc.ndim == 1:
                        sc = sc.reshape(-1, 1)
                    wq = wq * sc
                emb.weight = nn.Parameter(wq.to(emb.weight.dtype), requires_grad=False)
                model_state_dict.pop(f"{emb_prefix}.weight", None)
                print("[ConvRot] embed_tokens 已反量化 fp8 权重为 Embedding.weight。", file=sys.stderr)
            elif f"{emb_prefix}.weight_packed" in model_state_dict:
                # ---- W4A8 风格 embed ----
                packed = model_state_dict.pop(f"{emb_prefix}.weight_packed")
                s_rel = model_state_dict.pop(f"{emb_prefix}.weight_s_rel").float()
                s_channel = model_state_dict.pop(f"{emb_prefix}.weight_s_channel").float()
                codebook = model_state_dict.pop(f"{emb_prefix}.weight_codebook").float()
                model_state_dict.pop(f"{emb_prefix}.weight", None)
                # 反量化（与 ConvRotW4A8Linear._get_original_weight 一致）
                qint = _unpack_uint4_row_major(packed)        # unsigned 0..15 [V, in]
                V, in_f = qint.shape
                gs = in_f // s_rel.shape[1]                    # 由 s_rel 形状反推 group_size
                groups = s_rel.shape[1]
                group_scale = s_channel.reshape(-1, 1) * s_rel  # [V, groups]
                decoded = codebook[qint.long()].reshape(V, groups, gs)  # [V, groups, gs]
                w_rot = (decoded * group_scale.reshape(V, groups, 1)).reshape(V, in_f)
                # 逆 Hadamard（convrot_groupsize 块，256 优先，回退到能整除的值）
                cgs = 256
                if in_f % cgs != 0:
                    for cand in (64, 16):
                        if in_f % cand == 0:
                            cgs = cand
                            break
                h = _build_hadamard(cgs, device=w_rot.device, dtype=torch.float32)
                w = _rotate_weight(w_rot, h, cgs)               # original space
                emb.weight = nn.Parameter(w.to(emb.weight.dtype), requires_grad=False)
                print(f"[ConvRot] embed_tokens 已反量化 w4a8 权重为 Embedding.weight "
                      f"(group_size={gs}, convrot_groupsize={cgs})。", file=sys.stderr)
        except Exception as _e:
            print(f"[ConvRot] embed_tokens 反量化失败: {_e}", file=sys.stderr)

        # 顶层 scaled_fp8 marker 是标量标记，无对应模块参数，剔除避免 unexpected 噪音
        if "scaled_fp8" in model_state_dict:
            model_state_dict.pop("scaled_fp8")

        # LoRALinear keeps weight/bias compatible with nn.Linear but adds
        # lora_A/lora_B, which are absent from base pretrained checkpoints.
        _load_ret = model.load_state_dict(model_state_dict, strict=False)
        # 诊断：量化层反量化正常但输出异常时，多半是 non-quant 层权重缺失/多余
        # （strict=False 会静默丢弃），这里打印以便定位 checkpoint 完整性。
        try:
            _miss = list(_load_ret.missing_keys)
            _unexp = list(_load_ret.unexpected_keys)
            # print(f"[ConvRot-LOAD-DIAG] convrot_plan={len(convrot_plan)} "
            #       f"replaced={len(replaced_keys)} "
            #       f"missing={len(_miss)} unexpected={len(_unexp)}", file=sys.stderr)
            if _miss:
                print(f"[ConvRot-LOAD-DIAG] MISSING (first 20): {_miss[:20]}", file=sys.stderr)
            if _unexp:
                print(f"[ConvRot-LOAD-DIAG] UNEXPECTED (first 20): {_unexp[:20]}", file=sys.stderr)
        except Exception as _e:
            print(f"[ConvRot-LOAD-DIAG] err {_e}", file=sys.stderr)
        del model_state_dict, vae_state_dict
        if training:
            return model
        return model.to(model.device).eval().optimize(disable=not optimize)

    # ------------------------------------------------------------------ #
    # LoRA Weight Management
    # ------------------------------------------------------------------ #
    def _find_module_by_path(self, path: str):
        """通过点分隔路径查找模块（支持 _orig_mod 前缀）。"""
        clean = path.replace("._orig_mod.", ".")
        parts = clean.split('.')
        m = self
        for p in parts:
            if isinstance(m, nn.ModuleList):
                try:
                    m = m[int(p)]
                except (ValueError, IndexError):
                    return None
            else:
                m = getattr(m, p, None)
            if m is None:
                return None
        return m

    def load_lora_weights(self, lora_path: str, device: str = None):
        """
        Load LoRA weights from file.
        Supports both safetensors and pytorch formats.
        ConvRotLinear layers: loads lora_A/lora_B into buffer via load_lora_buffer()
        for dequant-time merge. LoRALinear layers: normal param copy.

        Args:
            lora_path: Checkpoint path (directory or .safetensors/.ckpt file)
            device: Target device, defaults to model's current device
        Returns:
            tuple: (loaded_keys, skipped_keys)
        """
        from pathlib import Path

        device = device or self.device
        lora_p = Path(lora_path)

        if lora_p.is_dir():
            safetensors_file = lora_p / "lora_weights.safetensors"
            ckpt_file = lora_p / "lora_weights.ckpt"
        else:
            safetensors_file = lora_p if lora_p.suffix == ".safetensors" else None
            ckpt_file = lora_p if lora_p.suffix in [".ckpt", ".pth"] else None

        if safetensors_file and safetensors_file.exists() and SAFETENSORS_AVAILABLE:
            state_dict = load_file(str(safetensors_file), device=device)
        elif ckpt_file and ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            raise FileNotFoundError(f"LoRA checkpoint not found. Expected either {safetensors_file} or {ckpt_file}")

        # 步骤1: 构建 ConvRot 量化层路径映射 {clean_path: module}
        convrot_paths = {}
        for mod_name, module in self.named_modules():
            if isinstance(module, _CONVROT_LAYER_TYPES):
                clean = mod_name.replace("._orig_mod.", ".")
                convrot_paths[clean] = module

        # 步骤2: 只处理 ConvRotLinear 的 LoRA key → load_lora_buffer
        # key 格式: <convrot_module_path>.lora_A / <convrot_module_path>.lora_B
        # convrot_paths 中的 key 是干净的模块路径（已去除 _orig_mod 前缀）
        # LoRA checkpoint 中的 key 也是相同格式，所以用精确匹配即可
        lora_buffer_count = 0
        keys_to_remove = set()
        unmatched_lora_keys = []
        for key in list(state_dict.keys()):
            if key.endswith('.lora_A'):
                base = key[:-len('.lora_A')]
                b_key = key[:-len('_A')] + '_B'
                if b_key not in state_dict:
                    continue
                lora_A = state_dict[key]
                lora_B = state_dict[b_key]
                r = lora_A.shape[0]  # lora_A: [r, in_features]

                # 仅精确匹配：LoRA checkpoint 的路径必须与 ConvRotLinear 的路径完全一致
                convrot_mod = convrot_paths.get(base)
                if convrot_mod is not None:
                    # 验证 LoRA shape 与模块 shape 匹配
                    if lora_A.shape[1] != convrot_mod.in_features or lora_B.shape[0] != convrot_mod.out_features:
                        print(f"[ConvRot] SKIP (shape mismatch): {base} lora_A={lora_A.shape[1]} module.in={convrot_mod.in_features} lora_B={lora_B.shape[0]} module.out={convrot_mod.out_features}", file=sys.stderr)
                        continue
                    alpha = self.lora_config.alpha if self.lora_config is not None else r
                    convrot_mod.load_lora_buffer(lora_A, lora_B, alpha, r)
                    lora_buffer_count += 1
                    keys_to_remove.add(key)
                    keys_to_remove.add(b_key)
                else:
                    unmatched_lora_keys.append(base)

        # 从 state_dict 中移除已加载到 buffer 的 key
        for k in keys_to_remove:
            state_dict.pop(k, None)

        if lora_buffer_count > 0:
            print(f"[ConvRot] Loaded LoRA buffer for {lora_buffer_count} ConvRotLinear layers.", file=sys.stderr)

        # 步骤3: 加载剩余的非 convrot LoRA weights (LoRALinear)
        model_params = dict(self.named_parameters())
        key_mapping = {}
        for k in model_params:
            clean = k.replace("._orig_mod.", ".")
            if clean != k:
                key_mapping[clean] = k

        loaded_keys, skipped_keys = [], []
        for key, value in state_dict.items():
            target_key = key if key in model_params else key_mapping.get(key)
            if target_key:
                model_params[target_key].data.copy_(value.to(device))
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

        return loaded_keys, skipped_keys

    def _iter_lora_modules(self):
        """Iterate over all LoRALinear modules and ConvRotLinear with active LoRA buffer."""
        from ..modules.layers.lora import LoRALinear
        seen = set()
        for module in self.modules():
            mid = id(module)
            if mid in seen:
                continue
            seen.add(mid)
            if isinstance(module, LoRALinear):
                yield module
            elif isinstance(module, _CONVROT_LAYER_TYPES) and module._has_lora_buffer:
                yield module

    def set_lora_enabled(self, enabled: bool):
        """Enable/disable all LoRA layers (both LoRALinear and ConvRotLinear)."""
        for module in self._iter_lora_modules():
            if isinstance(module, _CONVROT_LAYER_TYPES):
                module.set_lora_buffer_enabled(enabled)
            else:
                module.set_enabled(enabled)

    def reset_lora_weights(self):
        """Reset all LoRA weights (A: kaiming, B: zeros), effectively unloading LoRA."""
        for module in self._iter_lora_modules():
            if hasattr(module, 'reset_lora_parameters'):
                module.reset_lora_parameters()
            elif isinstance(module, _CONVROT_LAYER_TYPES):
                # ConvRot*Linear: just disable the LoRA buffer (data stays, but won't be applied)
                module.set_lora_buffer_enabled(False)

    def get_lora_state_dict(self) -> dict:
        """Get all LoRA parameters (lora_A/lora_B)."""
        return {name: param.data.clone() for name, param in self.named_parameters() if "lora_" in name}


def match_state_dict(meta_model, sd,show_num=10):

    meta_model_keys = set(meta_model.state_dict().keys())   
    state_dict_keys = set(sd.keys())

    # 打印匹配的键的数量
    matching_keys = meta_model_keys.intersection(state_dict_keys)
    print(f"Matching keys count: {len(matching_keys)}")
    
    # 打印不在 meta_model 中但在 state_dict 中的键（多余键）
    extra_keys = state_dict_keys - meta_model_keys
    if extra_keys:
        print(f"Extra keys in state_dict (not in meta_model): {len(extra_keys)}")
        for key in list(extra_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 打印不在 state_dict 中但在 meta_model 中的键（缺失键）
    missing_keys = meta_model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state_dict (not in state_dict): {len(missing_keys)}")
        for key in list(missing_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 如果需要，也可以打印部分匹配的键
    print(f"Sample matching keys: {list(matching_keys)[:5]}")

def load_gguf_checkpoint(gguf_checkpoint_path):

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
  
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        quant_type = tensor.tensor_type

        
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data) #tensor.data.copy()
 
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
        del tensor,weights
        if i > 0 and i % 1000 == 0:  # 每1000个tensor执行一次gc
            logger.info(f"Processed {i}tensors...")
            gc.collect()
    del reader
    gc.collect()
    return parsed_parameters

def set_gguf2meta_model(meta_model,model_state_dict,dtype,device):
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
    hf_quantizer = GGUFQuantizer(quantization_config=g_config)
    hf_quantizer.pre_quantized = True


    hf_quantizer._process_model_before_weight_loading(
        meta_model,
        device_map={"": device} if device else None,
        state_dict=model_state_dict
    )
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    x,y=load_model_dict_into_meta(
        meta_model, 
        model_state_dict, 
        hf_quantizer=hf_quantizer,
        device_map={"": device} if device else None,
        dtype=dtype
    )
    print(x,"offload_index")
    print(y,"state_dict_index")

    hf_quantizer._process_model_after_weight_loading(meta_model)

    
    del model_state_dict
    gc.collect()
    return meta_model.to(dtype=dtype)

import torch
import json
import torch.nn as nn
import torch.nn.functional as F
try:
    from comfy_kitchen.tensor.int8 import _build_hadamard, _rotate_weight
except ImportError:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

try:
    from comfy_kitchen.backends.eager.svdquant import _unpack_int4_row_major
except ImportError:  # pragma: no cover - fallback mirror of the int4 codec
    def _unpack_int4_row_major(packed):
        x32 = packed.to(torch.int32)
        lo = x32 & 0x0F
        hi = (x32 >> 4) & 0x0F
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def _unpack_uint4_row_major(packed):
    """Unsigned int4 (0..15) 解包，用于 W4A8 的 codebook 索引（索引必须 0..15，不能用符号位）。

    必须定义在 try/except 之外，否则当 comfy_kitchen 可正常 import 时（不进入 except），
    _unpack_uint4_row_major 不会被定义，运行时会 NameError。
    """
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int32)

class ConvRotLinear(nn.Module):
    """INT8 + ConvRot 量化的 Linear 层，内置 LoRA 支持。

    权重以 int8 + per-channel scale 存储，forward 时首次调用解量化 + 逆 Hadamard 旋转，
    然后缓存 bf16 权重复用。如果启用了 LoRA，在缓存权重基础上叠加 LoRA delta。
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # int8 量化权重
        self.register_buffer('weight_int8', torch.empty((out_features, in_features), dtype=torch.int8))
        # per-channel scale (out_features, 1)
        self.register_buffer('weight_scale', torch.empty((out_features, 1), dtype=torch.float32))
        self.convrot_groupsize = 256
        # 标记位：True 表示需要首次解量化
        self._need_dequant = True
        # LoRA 临时 buffer 标记
        self._has_lora_buffer = False

        # LoRA 启用/禁用控制（与 LoRALinear 接口一致）
        self._lora_enabled = True

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def _apply(self, fn, recurse=True):
        """Override _apply to protect int8/scale buffers from dtype conversion."""
        for key, param in self._parameters.items():
            if param is not None:
                self._parameters[key] = fn(param)
        for key, buf in self._buffers.items():
            if buf is not None:
                if key in ('weight_int8', 'weight_scale'):
                    self._buffers[key] = buf.to(device=fn(buf).device)
                else:
                    self._buffers[key] = fn(buf)
        return self

    def load_lora_buffer(self, lora_A_tensor, lora_B_tensor, alpha, r):
        """将 LoRA 权重存为临时 buffer，每次 forward 在 original space 合并。

        forward 流程:
          1. dequant: int8 × scale → rotated f32
          2. un-rotate: rotated → original
          3. 合并 LoRA delta (original space)
          4. F.linear(x, original_with_lora, bias)

        不缓存 merged 结果，确保每次 forward 都正确应用 LoRA。
        """
        dev = self.weight_int8.device
        self.register_buffer('_lora_A_buf', lora_A_tensor.contiguous().to(device=dev, dtype=torch.bfloat16))
        self.register_buffer('_lora_B_buf', lora_B_tensor.contiguous().to(device=dev, dtype=torch.bfloat16))
        self._lora_scaling = alpha / r
        self._has_lora_buffer = True
        self._need_dequant = True  # 强制下次 forward 执行 dequant + merge

    @torch.no_grad()
    def _get_original_weight(self):
        """解量化 int8 → rotated f32 → 逆旋转 → original space f32。"""
        weight_f32 = self.weight_int8.float() * self.weight_scale  # [out, in], rotated space
        gs = self.convrot_groupsize
        k = weight_f32.shape[1]
        if k % gs != 0:
            for candidate in (256, 64, 16):
                if k % candidate == 0:
                    gs = candidate
                    break
        h = _build_hadamard(gs, device=weight_f32.device, dtype=torch.float32)
        return _rotate_weight(weight_f32, h, gs)  # original space

    def set_lora_buffer_enabled(self, enabled: bool):
        """启用/禁用 ConvRotLinear 的 LoRA buffer（与 LoRALinear.set_enabled 接口一致）。"""
        self._lora_enabled = enabled

    def forward(self, x):
        # 每次 forward 都从 int8 解量化到 original space
        w = self._get_original_weight()  # [out, in], float32, original space
        
        # 如果有 LoRA 且已启用，在 original space 合并 delta
        if self._has_lora_buffer and self._lora_enabled:
            delta = (self._lora_B_buf @ self._lora_A_buf).to(dtype=w.dtype, device=w.device) * self._lora_scaling
            w = w + delta
        
        # 输入是 bf16，输出保持 bf16，避免下游层 dtype 不匹配
        out = F.linear(x, w.to(dtype=x.dtype), self.bias)
        return out

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, convrot_gs={self.convrot_groupsize}, '
                f'int8=True')


class ConvRotInt4Linear(nn.Module):
    """INT4 + ConvRot 量化的 Linear 层，内置 LoRA 支持（与 ConvRotLinear 接口一致）。

    权重以 int4（packed 到 int8, shape [out, in//2]） + per-channel scale 存储，
    forward 时解包 int4 -> 反量化 -> 逆 Hadamard 旋转，得到 original space 的 fp 权重，
    再在 original space 叠加 LoRA delta。
    参考 comfy-model-tools-main/convrot_loader.ConvRotInt4Linear 实现。
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # int4 量化权重：每字节打包两个 int4 nibble -> int8 [out, in//2]
        self.register_buffer('weight_packed', torch.empty((out_features, in_features // 2), dtype=torch.int8))
        # per-channel scale (out_features,) —— 与 convrot_loader 的 int4 格式一致
        self.register_buffer('weight_scale', torch.empty((out_features,), dtype=torch.float32))
        self.convrot_groupsize = 256
        self._need_dequant = True
        self._has_lora_buffer = False
        self._lora_enabled = True
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def _apply(self, fn, recurse=True):
        """Override _apply to protect packed/scale buffers from dtype conversion."""
        for key, param in self._parameters.items():
            if param is not None:
                self._parameters[key] = fn(param)
        for key, buf in self._buffers.items():
            if buf is not None:
                if key in ('weight_packed', 'weight_scale'):
                    self._buffers[key] = buf.to(device=fn(buf).device)
                else:
                    self._buffers[key] = fn(buf)
        return self

    def load_lora_buffer(self, lora_A_tensor, lora_B_tensor, alpha, r):
        """将 LoRA 权重存为临时 buffer，每次 forward 在 original space 合并（与 ConvRotLinear 一致）。"""
        dev = self.weight_packed.device
        self.register_buffer('_lora_A_buf', lora_A_tensor.contiguous().to(device=dev, dtype=torch.bfloat16))
        self.register_buffer('_lora_B_buf', lora_B_tensor.contiguous().to(device=dev, dtype=torch.bfloat16))
        self._lora_scaling = alpha / r
        self._has_lora_buffer = True
        self._need_dequant = True

    @torch.no_grad()
    def _get_original_weight(self):
        """解包 int4 -> 反量化 -> 逆旋转 -> original space f32。"""
        qint = _unpack_int4_row_major(self.weight_packed)                 # signed int [out, in]
        w_rot = qint.float() * self.weight_scale.reshape(-1, 1)           # [out, in], rotated space
        gs = self.convrot_groupsize
        k = w_rot.shape[1]
        if k % gs != 0:
            for candidate in (256, 64, 16):
                if k % candidate == 0:
                    gs = candidate
                    break
        h = _build_hadamard(gs, device=w_rot.device, dtype=torch.float32)
        return _rotate_weight(w_rot, h, gs)                              # -> original space

    def set_lora_buffer_enabled(self, enabled: bool):
        self._lora_enabled = enabled

    def forward(self, x):
        w = self._get_original_weight()  # [out, in], float32, original space
        if not getattr(self, "_fwd_diag_done", False):
            self._fwd_diag_done = True
            try:
                wn = w.float()
                # print(f"[ConvRot-FWD-DIAG] FP8 fmt={self._convrot_format} "
                #       f"w.mean={float(wn.mean()):.6f} w.absmax={float(wn.abs().max()):.4f} "
                #       f"w.nan={int(wn.isnan().sum())} w.numel={int(wn.numel())}",
                #       file=sys.stderr)
            except Exception as _e:
                print(f"[ConvRot-FWD-DIAG] FP8 err {_e}", file=sys.stderr)
        if self._has_lora_buffer and self._lora_enabled:
            delta = (self._lora_B_buf @ self._lora_A_buf).to(dtype=w.dtype, device=w.device) * self._lora_scaling
            w = w + delta
        out = F.linear(x, w.to(dtype=x.dtype), self.bias)
        return out

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, convrot_gs={self.convrot_groupsize}, '
                f'int4=True')


class ConvRotFp8Linear(nn.Module):
    """FP8 (scaled_fp8) 量化的 Linear 层，内置 LoRA 支持（与 ConvRotLinear 接口一致）。

    权重以 fp8_e4m3fn 存储 + per-tensor/per-channel scale (fp32) 反量化，
    forward 时解量化到 original space 的 fp 权重，再叠加 LoRA delta。
    磁盘格式与 comfy-model-tools-main/quant_fp8_scaled.py 完全一致：
      - {prefix}weight        : fp8_e4m3fn [out, in]
      - {prefix}scale_inverted : fp32 scalar（= 1/scale）
    参考 comfy-model-tools-main/fp8_scaled_loader.Fp8ScaledLinear 实现。
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # fp8 量化权重 [out, in]
        self.register_buffer('weight_fp8', torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn))
        # 逆 scale（fp32 [out, 1]）：本项目专用 quant_fp8_voxcpm 产出（= 1/scale，per-channel）。
        # 注意：必须注册为 [out,1] 而非标量！checkpoint 的 *.scale_inverted 是 per-channel [N,1]
        # 张量，若 buffer 是标量会导致 load_state_dict(strict=False) 因形状不匹配静默丢弃，
        # 反量化时使用空标量 -> 每通道共用错误 scale -> 权重全错 -> 推理空白/错误声音。
        self.register_buffer('scale_inverted', torch.empty((out_features, 1), dtype=torch.float32))
        # per-channel scale（fp32 [out, 1]）：ComfyUI 原生 scaled_fp8 格式
        #   （load_state_dict 要求形状严格匹配，故注册为 [out_features, 1]）
        self.register_buffer('scale_weight', torch.empty((out_features, 1), dtype=torch.float32))
        # 记录实际磁盘格式，反量化时据此精确选择 scale，避免歧义：
        #   "scaled_fp8" 原生格式 -> 用 scale_weight（= 除数 scale，w = q*scale）
        #   "fp8_voxcpm" 本项目格式 -> 用 scale_inverted（= 1/scale，w = q*(1/scale)）
        self._convrot_format = "scaled_fp8"
        self._has_lora_buffer = False
        self._lora_enabled = True
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def _apply(self, fn, recurse=True):
        """Override _apply to protect fp8/scale buffers from dtype conversion."""
        for key, param in self._parameters.items():
            if param is not None:
                self._parameters[key] = fn(param)
        for key, buf in self._buffers.items():
            if buf is not None:
                if key in ('weight_fp8', 'scale_inverted', 'scale_weight'):
                    self._buffers[key] = buf.to(device=fn(buf).device)
                else:
                    self._buffers[key] = fn(buf)
        return self

    def load_lora_buffer(self, lora_A_tensor, lora_B_tensor, alpha, r):
        """将 LoRA 权重存为临时 buffer，forward 时在 original space 合并（与 ConvRotLinear 一致）。"""
        dev = self.weight_fp8.device
        self.register_buffer('_lora_A_buf', lora_A_tensor.contiguous().to(device=dev, dtype=torch.bfloat16))
        self.register_buffer('_lora_B_buf', lora_B_tensor.contiguous().to(device=dev, dtype=torch.bfloat16))
        self._lora_scaling = alpha / r
        self._has_lora_buffer = True

    @torch.no_grad()
    def _get_original_weight(self):
        """fp8 -> f32 original space（scaled_fp8 无 Hadamard 旋转，直接乘 scale）。

        反量化语义严格对应导出脚本 quant_fp8_scaled.py：
            q = w / scale   (scale = amax(dim=1) / FP8_MAX, per-channel)
            w = q * scale
        即 scale_weight 存储的是「除数」scale，反量化等于 q * scale_weight。
        """
        w = self.weight_fp8.float()  # [out, in], f32, q
        if not getattr(self, "_diag_done", False):
            self._diag_done = True
            # try:
            #     print(f"[ConvRot-DIAG] FP8 fmt={self._convrot_format} "
            #           f"w_fp8.abs().sum={float(w.abs().sum()):.4f} "
            #           f"scale_inverted={float(self.scale_inverted.abs().sum()):.6f} "
            #           f"scale_weight.abs().sum={float(self.scale_weight.abs().sum()):.6f}",
            #           file=sys.stderr)
            # except Exception as _e:
            #     print(f"[ConvRot-DIAG] err {_e}", file=sys.stderr)
        if self._convrot_format == "fp8_voxcpm":
            # 本项目专用格式：scale_inverted = 1/scale，反量化 = q * (1/scale)
            if float(self.scale_inverted.abs().sum()) == 0.0:
                raise RuntimeError(
                    "[ConvRot] FP8 层反量化失败：scale_inverted 为全 0，说明该键未从 checkpoint 加载。"
                    "请确认 checkpoint 是本项目专用 fp8_voxcpm 格式（含逐层 .scale_inverted）。"
                )
            w = w * self.scale_inverted
        else:
            # 原生 scaled_fp8 格式：scale_weight = scale（除数），反量化 = q * scale
            if float(self.scale_weight.abs().sum()) == 0.0:
                raise RuntimeError(
                    "[ConvRot] FP8 层反量化失败：scale_weight 为全 0，说明该键未从 checkpoint 加载。"
                    "常见原因：导出时使用了 --scale-mode none（不保存 scale），或 checkpoint 缺少 .scale_weight 键。"
                    "请用 quant_fp8_scaled.py 默认 --scale-mode per-channel 重新导出。"
                )
            if self.scale_weight.ndim == 1:
                w = w * self.scale_weight.reshape(-1, 1)
            else:
                w = w * self.scale_weight
        return w

    def set_lora_buffer_enabled(self, enabled: bool):
        self._lora_enabled = enabled

    def forward(self, x):
        w = self._get_original_weight()  # [out, in], float32, original space
        if not getattr(self, "_fwd_diag_done", False):
            self._fwd_diag_done = True
            try:
                wn = w.float()
                # print(f"[ConvRot-FWD-DIAG] FP8 fmt={self._convrot_format} "
                #       f"w.mean={float(wn.mean()):.6f} w.absmax={float(wn.abs().max()):.4f} "
                #       f"w.nan={int(wn.isnan().sum())} w.numel={int(wn.numel())}",
                #       file=sys.stderr)
            except Exception as _e:
                print(f"[ConvRot-FWD-DIAG] FP8 err {_e}", file=sys.stderr)
        if self._has_lora_buffer and self._lora_enabled:
            delta = (self._lora_B_buf @ self._lora_A_buf).to(dtype=w.dtype, device=w.device) * self._lora_scaling
            w = w + delta
        out = F.linear(x, w.to(dtype=x.dtype), self.bias)
        return out

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, fp8=e4m3fn')


class ConvRotW4A8Linear(nn.Module):
    """W4A8 (asym_w4a8_int8) 量化的 Linear 层，内置 LoRA 支持（与 ConvRotLinear 接口一致）。

    权重以 int4（packed 到 int8） + per-group s_rel(fp8_e4m3fn/ fp32 [out, in//group_size])
    + per-channel s_channel(fp32 [out]) + codebook(fp32 [16]) 存储，激活侧由下游
    int8 动态量化完成（本层只负责权重反量化）。
    forward 时解包 int4 -> 反量化 -> 逆 Hadamard 旋转，得到 original space 的 fp 权重，
    再叠加 LoRA delta。
    磁盘格式与 comfy-model-tools-main/quant_w4a8_convrot.py 完全一致：
      - {prefix}weight          : int8 packed [out, in//2]（row-major，低 nibble=偶数列）
      - {prefix}weight_s_rel    : fp8/fp32 per-group scale [out, in//group_size]
      - {prefix}weight_s_channel: fp32 per-channel scale [out]
      - {prefix}weight_codebook : fp32 [16] Lloyd-Max levels
      - {prefix}comfy_quant     : {"format":"asym_w4a8_int8","group_size":G,"convrot_groupsize":C}
    反量化数学（与 comfy_kitchen / quant_w4a8_convrot 一致）：
      group_scale = s_channel[:,None] * s_rel        # [out, groups]
      w_rot = codebook[qint].reshape(out, groups, gs) * group_scale[:,:,None]   # [out, in]
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # int4 量化权重：每字节打包两个 int4 nibble -> int8 [out, in//2]（row-major）
        self.register_buffer('weight_packed', torch.empty((out_features, in_features // 2), dtype=torch.int8))
        # per-group relative scale。comfy-model-tools 默认 --scale_dtype fp32 导出为
        # float32；若用户用 --scale_dtype fp8 导出则为 fp8_e4m3fn。
        # 注意：务必用 fp32 注册！quant_w4a8_convrot.py 默认导出 s_rel 是 fp32，
        # 且 per-group s_rel = group_scale/s_channel 可能远小于 1，若以 fp8_e4m3fn 存储
        # 会下溢成 0 -> 整组权重反量化归零 -> 推理输出被压扁成空白。
        # 因此 buffer 统一 fp32；加载时若 checkpoint 为 fp8 则就地 .float() 转 fp32。
        self.register_buffer('weight_s_rel', torch.empty((out_features, in_features // 16), dtype=torch.float32))
        # per-channel scale (fp32 [out])
        self.register_buffer('weight_s_channel', torch.empty((out_features,), dtype=torch.float32))
        # codebook (fp32 [16])
        self.register_buffer('weight_codebook', torch.empty((16,), dtype=torch.float32))
        self.group_size = 16          # 与 comfy_quant 的 group_size 对应（反量化 reshape 用）
        self.convrot_groupsize = 256  # Hadamard 旋转块大小（与 comfy_quant 的 convrot_groupsize 对应）
        self._has_lora_buffer = False
        self._lora_enabled = True
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def _apply(self, fn, recurse=True):
        """Override _apply to protect packed/scale buffers from dtype conversion."""
        for key, param in self._parameters.items():
            if param is not None:
                self._parameters[key] = fn(param)
        for key, buf in self._buffers.items():
            if buf is not None:
                if key in ('weight_packed', 'weight_s_rel', 'weight_s_channel', 'weight_codebook'):
                    self._buffers[key] = buf.to(device=fn(buf).device)
                else:
                    self._buffers[key] = fn(buf)
        return self

    def load_lora_buffer(self, lora_A_tensor, lora_B_tensor, alpha, r):
        """将 LoRA 权重存为临时 buffer，forward 时在 original space 合并（与 ConvRotLinear 一致）。"""
        dev = self.weight_packed.device
        self.register_buffer('_lora_A_buf', lora_A_tensor.contiguous().to(device=dev, dtype=torch.bfloat16))
        self.register_buffer('_lora_B_buf', lora_B_tensor.contiguous().to(device=dev, dtype=torch.bfloat16))
        self._lora_scaling = alpha / r
        self._has_lora_buffer = True

    @torch.no_grad()
    def _get_original_weight(self):
        """解包 int4 -> 反量化 -> 逆旋转 -> original space f32。"""
        if not getattr(self, "_diag_done", False):
            self._diag_done = True
            try:
                print(f"[ConvRot-DIAG] W4A8 packed.abs().sum={float(self.weight_packed.float().abs().sum()):.4f} "
                      f"s_channel.abs().sum={float(self.weight_s_channel.abs().sum()):.6f} "
                      f"s_rel.abs().sum={float(self.weight_s_rel.float().abs().sum()):.6f} "
                      f"codebook.abs().sum={float(self.weight_codebook.abs().sum()):.6f}",
                      file=sys.stderr)
            except Exception as _e:
                print(f"[ConvRot-DIAG] W4A8 err {_e}", file=sys.stderr)
        qint = _unpack_uint4_row_major(self.weight_packed)                    # unsigned int 0..15 [out, in]
        out, k = qint.shape
        gs = self.group_size
        groups = k // gs
        # 反量化：group_scale = s_channel[:,None] * s_rel   -> [out, groups]
        group_scale = self.weight_s_channel.reshape(-1, 1) * self.weight_s_rel.float()  # [out, groups]
        # codebook 索引（qint 必须 0..15，故用 unsigned 解包）+ 重组为 [out, groups, gs]
        decoded = self.weight_codebook[qint.long()].reshape(out, groups, gs)   # [out, groups, gs]
        w_rot = (decoded * group_scale.reshape(out, groups, 1)).reshape(out, k)  # [out, in], rotated space
        # Hadamard 逆旋转（convrot_groupsize 块）
        cgs = self.convrot_groupsize
        if k % cgs != 0:
            for candidate in (256, 64, 16):
                if k % candidate == 0:
                    cgs = candidate
                    break
        h = _build_hadamard(cgs, device=w_rot.device, dtype=torch.float32)
        return _rotate_weight(w_rot, h, cgs)                                  # -> original space

    def set_lora_buffer_enabled(self, enabled: bool):
        self._lora_enabled = enabled

    def forward(self, x):
        w = self._get_original_weight()  # [out, in], float32, original space
        if not getattr(self, "_fwd_diag_done", False):
            self._fwd_diag_done = True
            try:
                wn = w.float()
                # print(f"[ConvRot-FWD-DIAG] W4A8 "
                #       f"w.mean={float(wn.mean()):.6f} w.absmax={float(wn.abs().max()):.4f} "
                #       f"w.nan={int(wn.isnan().sum())} w.numel={int(wn.numel())}",
                #       file=sys.stderr)
            except Exception as _e:
                print(f"[ConvRot-FWD-DIAG] W4A8 err {_e}", file=sys.stderr)
        if self._has_lora_buffer and self._lora_enabled:
            delta = (self._lora_B_buf @ self._lora_A_buf).to(dtype=w.dtype, device=w.device) * self._lora_scaling
            w = w + delta
        out = F.linear(x, w.to(dtype=x.dtype), self.bias)
        return out

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, group_size={self.group_size}, '
                f'convrot_gs={self.convrot_groupsize}, w4a8=True')


def _find_module_by_key(model, key):
    """通过点分隔的 key 在 module 树中查找子模块。"""
    parts = key.split('.')
    m = model
    for p in parts:
        if isinstance(m, nn.ModuleList):
            m = m[int(p)]
        else:
            m = getattr(m, p, None)
        if m is None:
            return None
    return m


_CONVROT_CLASS_BY_FMT = {
    "int8_tensorwise": ConvRotLinear,
    "convrot_w4a4": ConvRotInt4Linear,
    "scaled_fp8": ConvRotFp8Linear,
    "fp8_voxcpm": ConvRotFp8Linear,
    "asym_w4a8_int8": ConvRotW4A8Linear,
}

_CONVROT_LAYER_TYPES = (ConvRotLinear, ConvRotInt4Linear, ConvRotFp8Linear, ConvRotW4A8Linear)


def replace_linear_with_convrot(model, convrot_plan=None, clear_lora=True):
    """替换指定路径上的 nn.Linear 为量化 Linear 层。

    Args:
        model: 要替换的模型
        convrot_plan: {base_path: format} 需要替换的层基路径及格式。
            "int8_tensorwise"     -> ConvRotLinear（INT8 ConvRot）
            "convrot_w4a4"        -> ConvRotInt4Linear（INT4 ConvRot）
            "scaled_fp8"          -> ConvRotFp8Linear（FP8 scaled）
            "asym_w4a8_int8"      -> ConvRotW4A8Linear（W4A8 ConvRot）
            如果为 None，则替换所有 nn.Linear（旧行为，默认 int8）。
        clear_lora: 保留参数以兼容旧签名（当前不再需要）。
    """
    replaced_keys = set()
    if convrot_plan is not None:
        for base_key, fmt in convrot_plan.items():
            parts = base_key.split('.')
            parent = model
            for p in parts[:-1]:
                if isinstance(parent, nn.ModuleList):
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p, None)
                if parent is None:
                    break
            if parent is None:
                continue
            child_name = parts[-1]
            module = getattr(parent, child_name, None)
            if module is None:
                continue
            # 检查是否是 nn.Linear（排除已替换的 convrot 层）
            if isinstance(module, nn.Linear) and not isinstance(module, _CONVROT_LAYER_TYPES):
                cls = _CONVROT_CLASS_BY_FMT.get(fmt, ConvRotLinear)
                new_layer = cls(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None
                )
                setattr(parent, child_name, new_layer)
                replaced_keys.add(base_key)
    else:
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear):
                if isinstance(module, _CONVROT_LAYER_TYPES):
                    continue
                new_layer = ConvRotLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None
                )
                setattr(model, name, new_layer)
            else:
                replace_linear_with_convrot(module, convrot_plan=None)
    return model, replaced_keys


