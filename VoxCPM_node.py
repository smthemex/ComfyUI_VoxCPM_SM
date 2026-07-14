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
import soundfile as sf
from pathlib import PureWindowsPath
from safetensors import safe_open
import json
MAX_SEED = np.iinfo(np.int32).max
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

node_cr_path = os.path.dirname(os.path.abspath(__file__))


weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir

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
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),   
                io.Combo.Input("version",options= ["v2","v15",]  ),
                io.Combo.Input("lora",options= ["none"] + folder_paths.get_filename_list("loras") ), 
                io.Int.Input("lora_rank", default=32, min=8, max=128, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("lora_alpha", default=16, min=1, max=128, step=1, display_mode=io.NumberDisplay.number),
                io.Float.Input("lora_dropout", default=0.0, min=0.0, max=1.0, step=0.01, display_mode=io.NumberDisplay.number),
                io.Boolean.Input("enable_lm", default=True),
                io.Boolean.Input("enable_dit", default=True),
                io.Boolean.Input("enable_proj", default=False),
                io.Boolean.Input("denoise", default=False),
            ],
            outputs=[
                io.Custom("VoxCPM_SM_Model").Output("model"),
                ],
            )
    @classmethod
    def execute(cls, dit,gguf,vae,version,lora,lora_rank,lora_alpha,lora_dropout,enable_lm,enable_dit,enable_proj,denoise) -> io.NodeOutput:
        # Temporarily set environment variables to avoid CUDA graph issues
        os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"     
        def load_lora_config(safetensors_path):
            try:
                with safe_open(safetensors_path, framework="pt") as f:
                    # 获取metadata
                    metadata = f.metadata()
                    if "lora_config" in metadata:
                        lora_info = json.loads(metadata["lora_config"])
                        return  lora_info
                    else:
                        return  None
            except Exception as e:
                return None

        try:
            vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
            ckpt_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
            lora_path=folder_paths.get_full_path("loras", lora) if lora != "none" else None
            gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
            assert (ckpt_path is not None or gguf_path is not None) and vae_path is not None, "Please select a valid model and vae"
            zipenhancer_model_id=os.path.join(node_cr_path, "VoxCPM/speech_zipenhancer_ans_multiloss_16k_base")  
            params={"vae_path":vae_path,"ckpt_path":ckpt_path,"load_denoiser":denoise,"optimize":False,"zipenhancer_model_id":zipenhancer_model_id,"gguf_path": gguf_path}
            
            # 如果提供了lora路径，在初始化模型时传递lora_weights_path参数
            # 这样VoxCPM会自动创建LoRA配置并加载权重
            if lora_path is not None:
                params["lora_weights_path"] = lora_path
                # 使用用户提供的LoRA配置参数
                from .VoxCPM.src.voxcpm.model.voxcpm import LoRAConfig
                if lora_path.endswith(".safetensors") and load_lora_config(lora_path) is not None : # 新增lora的 metadata读取
                    lora_info = load_lora_config(lora_path)
                    lora_rank = int(lora_info.get("r", lora_rank))
                    lora_alpha = int(lora_info.get("alpha", lora_alpha))
                    lora_dropout = float(lora_info.get("dropout", lora_dropout))
                    enable_lm = bool(lora_info.get("enable_lm", enable_lm))
                    enable_dit = bool(lora_info.get("enable_dit", enable_dit))
                    enable_proj = bool(lora_info.get("enable_proj", enable_proj))           
                params["lora_config"] = LoRAConfig(
                    r=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    enable_lm=enable_lm,
                    enable_dit=enable_dit,
                    enable_proj=enable_proj,
                    max_grad_norm=1.0,
                )
                print(f"Loading LoRA with config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}, "
                      f"enable_lm={enable_lm}, enable_dit={enable_dit}, enable_proj={enable_proj}")
            
            repo=os.path.join(node_cr_path, "VoxCPM/VoxCPM2")  if version == "v2" else os.path.join(node_cr_path, "VoxCPM/VoxCPM15")
            model=VoxCPM.from_pretrained(repo,**params)
            model.version = version
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
                io.Combo.Input("version",options= ["v2","v15",]  ),    
                io.String.Input("train_manifest",multiline=False,default="train_data_example.jsonl"),
                io.Combo.Input("sample_rate",options= [16000,44100]),   
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
    def execute(cls, dit,vae,version,train_manifest,sample_rate,batch_size,grad_accum_steps,
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
            
            config_file = os.path.join(node_cr_path, "VoxCPM/conf/voxcpm_v2/voxcpm_finetune_lora_w.yaml") if version == "v2"  else os.path.join(node_cr_path, "VoxCPM/conf/voxcpm_v1.5/voxcpm_finetune_lora_w.yaml")
            
            
            if train_manifest:
                train_manifest = PureWindowsPath(train_manifest).as_posix()
                if not os.path.exists(train_manifest):
                    raise ValueError("Invalid train_manifest path")
            else:
                raise ValueError("Please input a local train_manifest")

            # 加载YAML配置
            yaml_args = load_yaml_config(config_file)
            yaml_args['pretrained_path'] = os.path.join(node_cr_path, "VoxCPM/VoxCPM15") if version!= "v2"  else os.path.join(node_cr_path, "VoxCPM/VoxCPM2")
            yaml_args['train_manifest'] = train_manifest

            # 添加时间戳到保存路径，避免覆盖之前的训练
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(folder_paths.models_dir, f"loras/finetune_lora_{timestamp}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            yaml_args['save_path'] = save_path
            print(f"{train_manifest} training checkpoints will be saved to: {save_path}")

            out_sample_rate = 0
            config_file = os.path.join(yaml_args['pretrained_path'], "config.json")
            if os.path.isfile(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    out_sr = cfg.get("audio_vae_config", {}).get("out_sample_rate")
                    if out_sr:
                        out_sample_rate = int(out_sr)
                except Exception:
                    pass

            log_dir = os.path.join(save_path, "voxcpm_logs")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            yaml_args['tensorboard'] = log_dir
            yaml_args['sample_rate'] = sample_rate
            yaml_args['batch_size'] = batch_size
            yaml_args['out_sample_rate'] = out_sample_rate
            yaml_args['max_grad_norm'] = 0.0
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
                'version': version,
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
                io.Int.Input("train_steps", default=2000, min=100, max=100000, step=100, display_mode=io.NumberDisplay.number),
                io.Int.Input("current_step", default=0, min=0, max=1000000, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("save_interval", default=500, min=100, max=100000, step=100, display_mode=io.NumberDisplay.number),
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
            version = config_data["version"]
            
            # 修改训练参数以支持分步训练
            yaml_args['start_step'] = current_step
            yaml_args['train_steps'] = train_steps
            yaml_args['num_iters'] = current_step + train_steps
            # 更新保存间隔
            yaml_args['save_interval'] = save_interval
            
            # 加载训练函数
            
            from .VoxCPM.scripts.train_voxcpm_finetune_w2 import train 

            # 执行训练
            print(f",Training version is: {version} ,starting Lora training from step {current_step} for {train_steps} steps...")
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
                io.String.Input("ref_text",multiline=True,default="reference text, used when a prompt audio is provided for better prosody matching."),
                io.String.Input("voice_design",multiline=True,default="A young woman, gentle and sweet voice"),
                io.String.Input("text",multiline=True,default="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech."),
                io.Int.Input("steps", default=10, min=1, max=10000,display_mode=io.NumberDisplay.number),
                io.Float.Input("retry_badcase_ratio_threshold", default=6.0, min=0, max=10.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("cfg", default=2.0, min=0.0, max=100.0,step=0.01,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("normalize", default=True),
                io.Boolean.Input("retry_badcase", default=True),
                io.Int.Input("retry_badcase_max_times", default=3, min=1, max=100,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("controllable_cloning", default=False),
                io.Boolean.Input("ultimate_clone", default=False),
                io.Boolean.Input("streaming", default=False),
                io.Boolean.Input("save_wav", default=True),
                io.Audio.Input("audio",optional=True),
                ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        ) 
    @classmethod
    def execute(cls, model,ref_text,voice_design,text,steps,retry_badcase_ratio_threshold,cfg,normalize,retry_badcase,retry_badcase_max_times,seed,
                controllable_cloning,ultimate_clone,streaming,save_wav,audio=None ) -> io.NodeOutput: 
        # Temporarily set environment variables to avoid CUDA graph issues
        os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))+f'seed_{seed}'
        # def set_seed(seed=42):
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        # set_seed(seed) # 在LLM中设置固定随机，但是因为llm并未开启贪心解码（temperature=0），所有不一定有效，仅方便多次推理抽卡
        try:
            #pre data
            if audio is not None: 
                audio_file = os.path.join(folder_paths.get_temp_directory(), f"audio_refer_temp{audio_file_prefix}.wav")
                try :
                    import soundfile as sf
                    audio_np = audio["waveform"].squeeze(0).cpu().numpy()
                    if audio_np.ndim == 2:
                        audio_np = audio_np.T  # [C, T] -> [T, C]
                    elif audio_np.ndim == 1:
                        pass # 单声道直接保存
                    sf.write(str(audio_file), audio_np, int(audio["sample_rate"]))
                except Exception as e:
                    print(e)
                    buff = io_lib.BytesIO() 
                    torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"],format="FLAC")
                    with open(audio_file, 'wb') as f:
                        f.write(buff.getbuffer())
            else:
                audio_file=None
                ref_text=None

            if not text or not text.strip():
                raise ValueError("Please input text")
            origin_text = text
            if voice_design:
                text =  f"({voice_design}) "+text
            sample_rate=model.tts_model.sample_rate
            version=model.version
            output_path=os.path.join(folder_paths.get_output_directory(), f"VoxCPM_{audio_file_prefix}_{ text[:2]}.wav")
            if streaming:
                chunks = []
                for chunk in model.generate_streaming(
                    text =text,# "Streaming text to speech is easy with VoxCPM!",
                    prompt_wav_path=audio_file,      # optional: path to a prompt speech for voice cloning
                    prompt_text=ref_text,          # optional: reference text
                    cfg_value=cfg,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
                    inference_timesteps=steps,   # LocDiT inference timesteps, higher for better result, lower for fast speed
                    normalize=normalize,           # enable external TN tool, but will disable native raw text support
                    denoise=False if model.denoiser is None else True,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
                    retry_badcase=retry_badcase,        # enable retrying mode for some bad cases (unstoppable)
                    retry_badcase_max_times=retry_badcase_max_times,  # maximum retrying times
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
                    seed=seed,
                    # supports same args as above
                ):
                    chunks.append(chunk)
                wav = np.concatenate(chunks)
            else:
                if controllable_cloning:
                    assert audio_file is not None, "Please input audio"
                    wav_c = model.generate(text=origin_text,reference_wav_path=audio_file,seed=seed)
                    output_clone_path=os.path.join(folder_paths.get_output_directory(), f"VoxCPM_{audio_file_prefix}_clone_{ text[:2]}.wav")
                    sf.write(output_clone_path, wav_c,sample_rate )

                    wav = model.generate(
                    text=text, # "(slightly faster, cheerful tone)This is a cloned voice with style control.",
                    reference_wav_path=output_clone_path,
                    cfg_value=cfg,
                    inference_timesteps=steps,
                    seed=seed
                     )
                elif ultimate_clone and version == "v2": # only for VoxCPM2
                    wav = model.generate(
                        text=origin_text,
                        prompt_wav_path=audio_file,
                        prompt_text=ref_text,
                        reference_wav_path=audio_file,
                        seed=seed # optional, for better simliarity 
                    )
                else:
                    wav = model.generate(
                        text=text,
                        prompt_wav_path=audio_file,      # optional: path to a prompt speech for voice cloning
                        prompt_text=ref_text,          # optional: reference text
                        cfg_value=cfg,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
                        inference_timesteps=steps,   # LocDiT inference timesteps, higher for better result, lower for fast speed
                        normalize=normalize,           # enable external TN tool, but will disable native raw text support
                        denoise=False if model.denoiser is None else True,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
                        retry_badcase=retry_badcase,        # enable retrying mode for some bad cases (unstoppable)
                        retry_badcase_max_times=retry_badcase_max_times,  # maximum retrying times
                        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
                        seed=seed
                                )
            if save_wav:
                sf.write(output_path, wav,sample_rate )
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
