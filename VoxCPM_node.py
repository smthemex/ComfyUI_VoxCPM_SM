 # !/usr/bin/env python
# -*- coding: UTF-8 -*-
import io as io_lib
import numpy as np
import torch
import os
from .VoxCPM.src.voxcpm.core import VoxCPM
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import random
import torchaudio
from pathlib import PureWindowsPath
MAX_SEED = np.iinfo(np.int32).max
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

node_cr_path = os.path.dirname(os.path.abspath(__file__))
original_torchinductor = os.environ.get("TORCHINDUCTOR_DISABLE_CUDAGRAPHS")
original_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")

class VoxCPM_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="VoxCPM_SM_Model",
            display_name="VoxCPM_SM_Model",
            category="VoxCPM_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] +folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),   
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras") ), 
                io.Int.Input("lora_rank", default=32, min=8, max=128, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("lora_alpha", default=16, min=1, max=128, step=1, display_mode=io.NumberDisplay.number),
                io.Float.Input("lora_dropout", default=0.0, min=0.0, max=1.0, step=0.01, display_mode=io.NumberDisplay.number),
                io.Boolean.Input("enable_lm", default=True),
                io.Boolean.Input("enable_dit", default=True),
                io.Boolean.Input("enable_proj", default=False),
                io.Boolean.Input("denoise", default=True),
            ],
            outputs=[
                io.Custom("VoxCPM_SM_Model").Output("model"),
                ],
            )
    @classmethod
    def execute(cls, dit,vae,lora,lora_rank,lora_alpha,lora_dropout,enable_lm,enable_dit,enable_proj,denoise) -> io.NodeOutput:
        # Temporarily set environment variables to avoid CUDA graph issues
        os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"     
        try:
            vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
            ckpt_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
            lora_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
            assert ckpt_path is not None and vae_path is not None,"Please select a valid model and vae"
           
            params={"vae_path":vae_path,"ckpt_path":ckpt_path,"denoise":denoise,"optimize":False}
            
            # 如果提供了lora路径，在初始化模型时传递lora_weights_path参数
            # 这样VoxCPM会自动创建LoRA配置并加载权重
            if lora_path is not None:
                params["lora_weights_path"] = lora_path
                # 使用用户提供的LoRA配置参数
                from .VoxCPM.src.voxcpm.model.voxcpm import LoRAConfig
                params["lora_config"] = LoRAConfig(
                    r=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    enable_lm=enable_lm,
                    enable_dit=enable_dit,
                    enable_proj=enable_proj,
                )
                print(f"Loading LoRA with config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}, "
                      f"enable_lm={enable_lm}, enable_dit={enable_dit}, enable_proj={enable_proj}")
            
            model=VoxCPM.from_pretrained(os.path.join(node_cr_path, "VoxCPM/VoxCPM15"),**params)
            
            if lora_path is not None:
                model.set_lora_enabled(True)
            return io.NodeOutput(model)
        finally:
            # Restore original environment variables
            if original_torchinductor is None:
                os.environ.pop("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", None)
            else:
                os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = original_torchinductor
                
            if original_alloc_conf is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original_alloc_conf


class VoxCPM_SM_LoraTrainerInit(io.ComfyNode):   
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxCPM_SM_LoraTrainerInit",
            display_name="VoxCPM_SM_LoraTrainerInit",
            category="VoxCPM_SM",
            inputs=[
                io.Combo.Input("dit",options= ["none"] +folder_paths.get_filename_list("diffusion_models") ),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),     
                io.String.Input("train_manifest",multiline=False,default="train_data_example.jsonl"),
                io.Combo.Input("sample_rate",options= [44100,48000,96000,22050,24000,32000,192000]),   
                io.Int.Input("batch_size", default=1, min=1, max=1024,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("grad_accum_steps", default=1, min=1, max=1024,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("log_interval", default=10, min=1, max=1000000,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("valid_interval", default=1000, min=10, max=MAX_SEED,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("learning_rate", default= 0.0001, min=0.0, max=1.0,step=0.00001,display_mode=io.NumberDisplay.number),
                io.Float.Input("weight_decay", default= 0.01, min=0.0, max=1.0,step=0.001,display_mode=io.NumberDisplay.number),
                io.Int.Input("warmup_steps", default= 100, min=1, max=1000000,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("max_steps", default= 2000, min=1, max=1000000,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("max_batch_tokens", default= 0, min=0, max=MAX_SEED,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("lora_rank", default= 32, min=16, max=64,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("lora_alpha", default= 16, min=8, max=64,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("lora_dropout", default= 0.0, min=0.0, max=1,step=1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("enable_lm", default=True),
                io.Boolean.Input("enable_dit", default=True),
                io.Boolean.Input("enable_proj", default=False),
                ],
            outputs=[
                io.String.Output(display_name="info"),
                io.String.Output(display_name="config_path"),
            ],
        ) 
    
    @classmethod
    def execute(cls, dit,vae,train_manifest,sample_rate,batch_size,grad_accum_steps,
                log_interval,valid_interval,learning_rate,
                weight_decay,warmup_steps,max_steps,max_batch_tokens,lora_rank, lora_alpha,lora_dropout,
                enable_lm,enable_dit,enable_proj) -> io.NodeOutput: 
        # Temporarily set environment variables to avoid CUDA graph issues
        os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        try:
            from .VoxCPM.src.voxcpm.training.config import load_yaml_config
            
            vae_path = folder_paths.get_full_path("vae", vae) if vae != "none" else None
            ckpt_path = folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
            
            config_file = os.path.join(node_cr_path, "VoxCPM/conf/voxcpm_v1.5/voxcpm_finetune_lora_w.yaml")
            
            if train_manifest:
                train_manifest = PureWindowsPath(train_manifest).as_posix()
                if not os.path.exists(train_manifest):
                    raise ValueError("Invalid train_manifest path")
            else:
                raise ValueError("Please input a local train_manifest")

            # 加载YAML配置
            yaml_args = load_yaml_config(config_file)
            yaml_args['pretrained_path'] = os.path.join(node_cr_path, "VoxCPM/VoxCPM15")
            yaml_args['train_manifest'] = train_manifest

            # 添加时间戳到保存路径，避免覆盖之前的训练
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(folder_paths.models_dir, f"loras/finetune_lora_{timestamp}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            yaml_args['save_path'] = save_path
            print(f"Training checkpoints will be saved to: {save_path}")

            log_dir = os.path.join(save_path, "voxcpm_logs")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            yaml_args['tensorboard'] = log_dir
            yaml_args['sample_rate'] = sample_rate
            yaml_args['batch_size'] = batch_size
            yaml_args['grad_accum_steps'] = grad_accum_steps
            yaml_args['num_workers'] = 0
            # num_iters由loop节点控制，这里不设置
            yaml_args['log_interval'] = log_interval
            yaml_args['valid_interval'] = valid_interval
            # save_interval由loop节点控制，这里设置一个默认值
            yaml_args['save_interval'] = 1000  # 默认保存间隔
            yaml_args['learning_rate'] = learning_rate
            yaml_args['weight_decay'] = weight_decay
            yaml_args['warmup_steps'] = warmup_steps
            yaml_args['max_steps'] = max_steps
            yaml_args['max_batch_tokens'] = max_batch_tokens  # 0禁用过滤
            yaml_args["lora"]['r'] = lora_rank 
            yaml_args["lora"]['alpha'] = lora_alpha
            yaml_args["lora"]['dropout'] = lora_dropout
            yaml_args["lora"]['enable_lm'] = enable_lm
            yaml_args["lora"]['enable_dit'] = enable_dit
            yaml_args["lora"]['enable_proj'] = enable_proj
            
            # 保存配置到文件
            import json
            config_data = {
                'yaml_args': yaml_args,
                'ckpt_path': ckpt_path,
                'vae_path': vae_path,
                'save_path': save_path,
            }
            
            config_file_path = os.path.join(save_path, "training_config.json")
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            info = f"Training initialized. Config saved to: {config_file_path}"
            return io.NodeOutput(info, config_file_path)
        finally:
            # Restore original environment variables
            if original_torchinductor is None:
                os.environ.pop("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", None)
            else:
                os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = original_torchinductor
                
            if original_alloc_conf is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original_alloc_conf

class VoxCPM_SM_LoraTrainerLoop(io.ComfyNode):   
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxCPM_SM_LoraTrainerLoop",
            display_name="VoxCPM_SM_LoraTrainerLoop",
            category="VoxCPM_SM",
            inputs=[
                io.String.Input("config_path", multiline=False, default=""),
                io.Int.Input("train_steps", default=100, min=1, max=10000, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("current_step", default=0, min=0, max=1000000, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("save_interval", default=1000, min=10, max=100000, step=1, display_mode=io.NumberDisplay.number),
                ],
            outputs=[
                io.String.Output(display_name="info"),
                io.String.Output(display_name="checkpoint_path"),
                io.Int.Output(display_name="next_step"),
            ],
        ) 
    
    @classmethod
    def execute(cls, config_path, train_steps, current_step, save_interval) -> io.NodeOutput: 
        # Temporarily set environment variables to avoid CUDA graph issues
        os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        try:
            if not config_path or not os.path.exists(config_path):
                raise ValueError("Invalid config path. Please run VoxCPM_SM_LoraTrainerInit first.")
            
            # 加载配置
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            yaml_args = config_data['yaml_args']
            ckpt_path = config_data['ckpt_path']
            vae_path = config_data['vae_path']
            save_path = config_data['save_path']
            
            # 修改训练参数以支持分步训练
            yaml_args['start_step'] = current_step
            yaml_args['train_steps'] = train_steps
            yaml_args['num_iters'] = current_step + train_steps
            # 更新保存间隔
            yaml_args['save_interval'] = save_interval
            
            # 加载训练函数
            from .VoxCPM.scripts.train_voxcpm_finetune_w import train
            
            # 执行训练
            print(f"Starting Lora training from step {current_step} for {train_steps} steps...")
            with torch.inference_mode(False):
                # 现在train函数支持分步训练
                train(**yaml_args, ckpt_path=ckpt_path, vae_path=vae_path)
            
            # 计算下一步
            next_step = current_step + train_steps
            checkpoint_dir = os.path.join(save_path, f"step_{next_step:07d}")
            
            info = f"Training completed {train_steps} steps. Next step: {next_step}"
            return io.NodeOutput(info, checkpoint_dir, next_step)
        finally:
            # Restore original environment variables
            if original_torchinductor is None:
                os.environ.pop("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", None)
            else:
                os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = original_torchinductor
                
            if original_alloc_conf is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original_alloc_conf

class VoxCPM_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxCPM_SM_KSampler",
            display_name="VoxCPM_SM_KSampler",
            category="VoxCPM_SM",
            inputs=[
                io.Custom("VoxCPM_SM_Model").Input("model"),  
                io.String.Input("prompt",multiline=True,default="reference text, used when a prompt audio is provided for better prosody matching."),
                io.String.Input("text",multiline=True,default="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech."),
                io.Int.Input("steps", default=10, min=1, max=10000,display_mode=io.NumberDisplay.number),
                io.Float.Input("retry_badcase_ratio_threshold", default=6.0, min=0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("cfg", default=2.0, min=0.0, max=100.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("normalize", default=True),
                io.Boolean.Input("retry_badcase", default=True),
                io.Int.Input("retry_badcase_max_times", default=3, min=1, max=100,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("streaming", default=False),
                io.Boolean.Input("save_wav", default=True),
                io.Audio.Input("audio",optional=True),
                ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        ) 
    @classmethod
    def execute(cls, model,prompt,text,steps,retry_badcase_ratio_threshold,cfg,normalize,retry_badcase,retry_badcase_max_times,streaming,save_wav,audio=None ) -> io.NodeOutput: 
        # Temporarily set environment variables to avoid CUDA graph issues
        os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        try:
            #pre data
            if audio is not None:          
                audio_file = os.path.join(folder_paths.get_temp_directory(), f"audio_refer_temp{audio_file_prefix}.wav")
                buff = io_lib.BytesIO() 
                torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"],format="FLAC")
                with open(audio_file, 'wb') as f:
                    f.write(buff.getbuffer())
            else:
                audio_file=None
                prompt=None

            if not text or not text.strip():
                raise ValueError("Please input text")
            
            if streaming:
                chunks = []
                for chunk in model.generate_streaming(
                    text =text,# "Streaming text to speech is easy with VoxCPM!",
                    prompt_wav_path=audio_file,      # optional: path to a prompt speech for voice cloning
                    prompt_text=prompt,          # optional: reference text
                    cfg_value=cfg,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
                    inference_timesteps=steps,   # LocDiT inference timesteps, higher for better result, lower for fast speed
                    normalize=normalize,           # enable external TN tool, but will disable native raw text support
                    denoise=False if model.denoiser is None else True,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
                    retry_badcase=retry_badcase,        # enable retrying mode for some bad cases (unstoppable)
                    retry_badcase_max_times=retry_badcase_max_times,  # maximum retrying times
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
                    # supports same args as above
                ):
                    chunks.append(chunk)
                wav = np.concatenate(chunks)
            else:
                wav = model.generate(
                    text=text,
                    prompt_wav_path=audio_file,      # optional: path to a prompt speech for voice cloning
                    prompt_text=prompt,          # optional: reference text
                    cfg_value=cfg,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
                    inference_timesteps=steps,   # LocDiT inference timesteps, higher for better result, lower for fast speed
                    normalize=normalize,           # enable external TN tool, but will disable native raw text support
                    denoise=False if model.denoiser is None else True,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
                    retry_badcase=retry_badcase,        # enable retrying mode for some bad cases (unstoppable)
                    retry_badcase_max_times=retry_badcase_max_times,  # maximum retrying times
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
                        )
            sample_rate=model.tts_model.sample_rate
            if save_wav:
                import soundfile as sf
                sf.write(os.path.join(folder_paths.get_output_directory(), f"VoxCPM_{audio_file_prefix}_{ text[:2]}.wav"), wav,sample_rate )
            waveform = torch.from_numpy(wav).unsqueeze(0) #torch.Size([1, 232848])
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate":sample_rate}
            return io.NodeOutput(audio)
        finally:
            # Restore original environment variables
            if original_torchinductor is None:
                os.environ.pop("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", None)
            else:
                os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = original_torchinductor
                
            if original_alloc_conf is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original_alloc_conf



from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/VoxCPM_SM_Extension")
async def get_hello(request):
    return web.json_response("VoxCPM_SM_Extension")

class VoxCPM_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VoxCPM_SM_Model,
            VoxCPM_SM_LoraTrainerInit,
            VoxCPM_SM_LoraTrainerLoop,
            VoxCPM_SM_KSampler,
        ]


async def comfy_entrypoint() -> VoxCPM_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return VoxCPM_SM_Extension()
