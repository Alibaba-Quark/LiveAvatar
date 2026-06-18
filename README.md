<div align="center">

<p align="center">
  <img src="./assets/logo.png" width="200px" alt="Live Avatar Teaser">
</p>

<h1>🎬 Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length</h1>
<!-- <h3>The code will be open source in <strong><span style="color: #87CEEB;">early December</span></strong>.</h3> -->



<p>
<a href="https://github.com/Yubo-Shankui" style="color: inherit;">Yubo Huang</a><sup>1,2</sup> ·
<a href="#" style="color: inherit;">Hailong Guo</a><sup>2,3</sup> ·
<a href="#" style="color: inherit;">Fangtai Wu</a><sup>2,4</sup> ·
<a href="#" style="color: inherit;">Weiqiang Wang</a><sup>5</sup> ·
<a href="#" style="color: inherit;">Shifeng Zhang</a><sup>2</sup> ·
<a href="#" style="color: inherit;">Shijie Huang</a><sup>2</sup> ·
<a href="#" style="color: inherit;">Qijun Gan</a><sup>4</sup> ·
<a href="#" style="color: inherit;">Lin Liu</a><sup>1</sup> ·
<a href="#" style="color: inherit;">Ruihua Huang</a><sup>2,*</sup> ·
<a href="#" style="color: inherit;">Sirui Zhao</a><sup>1,*</sup> ·
<a href="http://staff.ustc.edu.cn/~cheneh/" style="color: inherit;">Enhong Chen</a><sup>1,*</sup> ·
<a href="https://openreview.net/profile?id=%7EJiaming_Liu7" style="color: inherit;">Jiaming Liu</a><sup>2,*,‡</sup> ·
<a href="https://sites.google.com/view/stevenhoi/" style="color: inherit;">Steven Hoi</a><sup>2</sup>
</p>

<p style="font-size: 0.9em;">
<sup>1</sup> University of Science and Technology of China &nbsp;&nbsp;
<sup>2</sup> Alibaba Group &nbsp;&nbsp;
<sup>3</sup> Beijing University of Posts and Telecommunications &nbsp;&nbsp;
<sup>4</sup> Zhejiang University
<sup>5</sup> Monash University
  
</p>

<p style="font-size: 0.9em;">
<sup>*</sup> Corresponding authors. &nbsp;&nbsp; <sup>‡</sup> Project leader.
</p>

<!-- Badges -->
<a href="https://arxiv.org/abs/2512.04677"><img src="https://img.shields.io/badge/arXiv-2512.04677-b31b1b.svg?style=for-the-badge" alt="arXiv"></a> <a href="https://huggingface.co/papers/2512.04677"><img src="https://img.shields.io/badge/🤗%20Daily%20Paper-ff9d00?style=for-the-badge" alt="Daily Paper"></a> <a href="https://huggingface.co/Quark-Vision/Live-Avatar"><img src="https://img.shields.io/badge/Hugging%20Face-Model-ffbd45?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"></a> <a href="https://github.com/Alibaba-Quark/LiveAvatar"><img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github"></a> <a href="https://liveavatar.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>

</div>

> **TL;DR:** **Live Avatar** is an algorithm–system co-designed framework that enables real-time, streaming, infinite-length interactive avatar video generation. Powered by a **14B-parameter** diffusion model, it achieves **45 FPS** on multi-card **H800** GPUs with **4-step** sampling and supports **Block-wise Autoregressive** processing for **10,000+** second streaming videos.

<div align="center">

[![Watch the video](assets/demo.png)](https://www.youtube.com/watch?v=srbsGlLNpAc)

<strong>👀 More Demos:</strong> <br>
🤖 Human-AI Conversation &nbsp;|&nbsp; ♾️ Infinite Video &nbsp;|&nbsp; 🎭 Diverse Characters &nbsp;|&nbsp; 🎬 Animated Tech Explanation <br>
<a href="https://liveavatar.github.io/">
  <strong>👉 Click Here to Visit Project Page! 🌐</strong>
</a>
<br>

</div>

---
## ✨ Highlights

> - ⚡ **​​Real-time Streaming Interaction**​​ - Achieve **45** FPS real-time streaming with low latency
> - ♾️ ​​**​​Infinite-length Autoregressive Generation**​​​​ - Support **10,000+** second continuous video generation
> - 🎨 ​​**​​Generalization Performances**​​​​ - Strong generalization across cartoon characters, singing, and diverse scenarios 


---
## 📰 News
- **[2026.6.18]** 🎉 LiveAvatar has been accepted by **ECCV 2026**! 🥳
- **[2026.1.20]** 🚀 Major performance breakthrough (**v1.1**)! **FP8 quantization** enables inference on **48GB GPUs**, while advanced **compilation** and **cuDNN** attention boost speed to **~2.5x** peak and **3x** average FPS. Achieving stable **45+ FPS** on multi-H800 — share your results on different GPUs! Inference fixes also bring noticeable **quality improvements**, significantly surpassing the teacher model on qualitative metrics.
<!-- - **[2026.1.9]** 🚀 Major performance update! Inference speed boosted to Peak 1.5x and Average 2x, achieving stable 30+ FPS on multi-H800 setups.  -->
- **[2025.12.16]** 🎉 LiveAvatar has reached **1,000+** stars on GitHub! Thank you to the community for the incredible support! ⭐
- **[2025.12.12]** 🚀 We released **single-gpu** inference [Code](infinite_inference_single_gpu.sh) — no need for 5×H800 (house-priced server), a single 80GB VRAM GPU is enough to enjoy. 
- **[2025.12.08]** 🚀 We released **real-time** inference [Code](infinite_inference_multi_gpu.sh) and the model [Weight](https://huggingface.co/Quark-Vision/Live-Avatar).
- **[2025.12.08]** 🎉 LiveAvatar won the Hugging Face [#1 Paper of the day](https://huggingface.co/papers/date/2025-12-05)!
- **[2025.12.04]** 🏃‍♂️ We committed to open-sourcing the code in **early December**.
- **[2025.12.04]** 🔥 We released [Paper](https://arxiv.org/abs/2512.04677) and [demo page](https://liveavatar.github.io/) Website.

---

## 📑 Todo List

### 🌟 **Early December** (core code release)

- ✅ Release the paper
- ✅ Release the demo website
- ✅ Release checkpoints on Hugging Face
- ✅ Release Gradio Web UI
- ✅ Experimental real-time streaming inference on at least H800 GPUs
  - ✅ Distribution-matching distillation to 4 steps
  - ✅ Timestep-forcing pipeline parallelism

### ⚙️ **Later updates**

- ✅ Inference code supporting single GPU (offline generation)
- ✅ Multi-character support
- ✅ Inference Acceleration Stage1 (RoPE optimization, compilation, LoRA merge)
- ✅ Streaming-VAE intergration
- ✅ Inference Acceleration Stage2 (further compilation, fp8, cudnn attn)
- ⬜ UI integration for easily streaming interaction
- ⬜ TTS integration
- ⬜ Training code 
- ⬜ LiveAvatar v1.2

## 🛠️ Installation

Please follow the steps below to set up the environment.

### 1. Create Environment
```bash
conda create -n liveavatar python=3.10 -y
conda activate liveavatar
```

### 2. Install CUDA Dependencies (optional)
```bash
conda install nvidia/label/cuda-12.4.1::cuda -y
conda install -c nvidia/label/cuda-12.4.1 cudatoolkit -y
```

### 3. Install PyTorch & Flash Attention
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# If you are using NVIDIA Hopper architecture (H800/H200, etc.), FlashAttention 3 is recommended for a significant speedup:
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 --extra-index-url https://download.pytorch.org/whl/cu128

# Otherwise, use FlashAttention 2:
pip install flash-attn==2.8.3 --no-build-isolation
```

### 4. Install Python Requirements
```bash
pip install -r requirements.txt
```
### 5. Install FFMPEG
```bash
apt-get update && apt-get install -y ffmpeg                 
```

---

## 📥 Download Models

Please download the pretrained checkpoints from links below and place them in the `./ckpt/` directory.

| Model Component | Description | Link |
| :--- | :--- | :---: |
| `WanS2V-14B` | base model| 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) |
| `liveAvatar` | our lora model| 🤗 [Huggingface](https://huggingface.co/Quark-Vision/Live-Avatar) |
```bash
# If you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./ckpt/Wan2.2-S2V-14B
huggingface-cli download Quark-Vision/Live-Avatar --local-dir ./ckpt/LiveAvatar
```

After downloading, your directory structure should look like this:

```
ckpt/
├── Wan2.2-S2V-14B/          # Base model
│   ├── config.json
│   ├── diffusion_pytorch_model-*.safetensors
│   └── ...
└── LiveAvatar/              # Our LoRA model
    ├── liveavatar.safetensors
    └── ...
```



## 🚀 Inference
### Real-time Inference with TPP
> 💡 Currently, This command can run on GPUs with at least 80GB VRAM.
```bash
# CLI Inference
bash infinite_inference_multi_gpu.sh
# Gradio Web UI
bash gradio_multi_gpu.sh
```
> 💡 The model can generate videos from audio input combined with reference image and optional text prompt.

> 💡 The `size` parameter represents the area of the generated video, with the aspect ratio following that of the original input image.

> 💡 The `--num_clip` parameter controls the number of video clips generated, useful for quick preview with shorter generation time.

> 💡 Currently, our TPP pipeline requires **five** GPUs for inference. We are planning to develop a 3-step version that can be deployed on a 4-GPU cluster.
Furthermore, we are planning to integrate the [LightX2V](https://github.com/ModelTC/LightX2V) VAE component. This integration will eliminate the dependency on additional single-GPU VAE parallelism and support 4-step inference within a 4-GPU setup.

> 💡 Compilation **(`ENABLE_COMPILE`)**: Enabling compilation will cause a long wait time during the first inference as the model compiles, but subsequent runs will see significant performance improvements. This is highly valuable for streaming long video scenarios. However, if you just want to quickly run a few test cases, we recommend disabling it by setting `export ENABLE_COMPILE=false` in your inference script.

> 💡 FP8 Quantization **(`ENABLE_FP8`)**: FP8 offers **notable VRAM savings**, enabling inference on **48GB GPUs**, and also provides modest performance gains. Note that this may cause slight quality degradation. You can enable it by setting `export ENABLE_FP8=true` in your inference script.

Please visit our [project page](https://liveavatar.github.io/) to see more examples and learn about the scenarios suitable for this model.
### Single-GPU Inference
> 💡 This command can run on a single GPU with at least 80GB VRAM.
```bash
# CLI Inference
bash infinite_inference_single_gpu.sh
# Gradio Web UI
bash gradio_single_gpu.sh
```

> 💡 If you encounter OOM errors after multiple runs in the Gradio Web UI, please try lowering the resolution (the `size` parameter) as a temporary fix. We are actively developing enhanced single GPU memory optimization; track our progress in the "Later updates" section.

> 💡 To avoid performance degradation caused by frequent CPU offloading, we set the `enable_online_decode` parameter to `false` by default in the single-GPU scripts. This may slightly reduce quality when generating extremely long videos; in such cases, consider adding `--enable_online_decode` to your inference command.
## 📝 Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@misc{huang2025liveavatarstreamingrealtime,
      title={Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length}, 
      author={Yubo Huang and Hailong Guo and Fangtai Wu and Shifeng Zhang and Shijie Huang and Qijun Gan and Lin Liu and Sirui Zhao and Enhong Chen and Jiaming Liu and Steven Hoi},
      year={2025},
      eprint={2512.04677},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.04677}, 
}
```
## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Alibaba-Quark/LiveAvatar&type=date&legend=top-left)](https://www.star-history.com/#Alibaba-Quark/LiveAvatar&type=date&legend=top-left)

## 📜 License Agreement
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](LICENSE).
* The Wan model (Our base model) is also released under the Apache 2.0 license as found in the [LICENSE](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt).
* The project is a research preview. Please contact us if you find any potential violations. (jmliu1217@gmail.com)

### 💬 WeChat Group

<p align="center">
  <img src="assets/wechat_group.png" alt="WeChat group" width="360" />
</p>



## 🙏 Acknowledgements

We would like to express our gratitude to the following projects:

*   [CausVid](https://github.com/tianweiy/CausVid)
*   [Longlive](https://github.com/NVlabs/LongLive)
*   [WanS2V](https://humanaigc.github.io/wan-s2v-webpage/)
