# ComfyUI_VoxCPM_SM

[VoxCPM](https://github.com/OpenBMB/VoxCPM)：  
Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning，you can use this node,easy infer and easy train

-----

# TIPS

* Lora train jsonl files in ' ./VoxCPM/examples/train_data_example.jsonl ' , edit it ,add audio and audio text ;  
* Lora训练的train\_data\_example.jsonl文件在' ./VoxCPM/examples/train_data_example.jsonl '里，按格式填写你的本地音频路径和对应的文本即可，有4种格式，刚开始按最简单的来就好；
* 5-10min train audios ，train 2000 step 
* 官方推荐的是5-10分钟音频数据，训练2000步即可 ，模型自动存在lora目录下； 训练节点只是懒得开webui搓的单线程，性能不是最优， 因为ComfyUI的天然缺陷在那。
* 训练方法，在init节点填写本地train_data_example.jsonl路径，去掉引号，选择模型和vae，点击运行；
* Training method: Fill in the local ‘ train_data_example.jsonl ’ path in the init node, remove the path quotation marks, select the model and VAE, and click Run;

# 1.Installation  

* In the ' ./ComfyUI/custom_nodes ' directory, run the following:   

```
git clone https://github.com/smthemex/ComfyUI_VoxCPM_SM

```

# 2.requirements  

```
pip install -r requirements.txt
```

# 3.checkpoints 

* 3.1 Vae and model [VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5)  只下载vae和models 模型   

```
├── ComfyUI/models/diffusion_models
|     ├── VoxCPM.safetensors   #  rename from  model.safetensors 换个名字
├── ComfyUI/models/vae
|     ├──audiovae.pth
```

# 4.Example
* Infer or train  lora 推理或训练lora  
![](https://github.com/smthemex/ComfyUI\_VoxCPM\_SM/blob/main/example\_workflows/example.png)

# 5.Citation
```
@article{voxcpm2025,
  title        = {VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning},
  author       = {Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and Li, Xiang and Yu, Renjie and Wang, Ziyang and Ye, Runchuan and Sun, Weiyue and Gui, Jiancheng and Li, Kehan and Wu, Zhiyong  and Liu, Zhiyuan},
  journal      = {arXiv preprint arXiv:2509.24650},
  year         = {2025},
}
```



