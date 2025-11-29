<div align="center">

# 👤 Live Avatar

**Streaming Real-time Infinite Inference for Interactive Avatars**

<!-- Badges -->
<a href="https://arxiv.org/abs/YOUR_PAPER_ID"><img src="https://img.shields.io/badge/arXiv-25XX.XXXXX-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
<a href="https://huggingface.co/YOUR_ORG/YOUR_MODEL"><img src="https://img.shields.io/badge/Hugging%20Face-Model-ffbd45?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"></a>
<a href="YOUR_DEMO_LINK"><img src="https://img.shields.io/badge/Demo-Gradio-orange?style=for-the-badge" alt="Demo"></a>
<a href="https://github.com/YOUR_USERNAME/LiveAvatar"><img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github"></a>


<!-- Teaser Video/GIF -->
<br>
<img src="./assets/liveavatar1.jpg" width="800px" alt="Live Avatar Teaser">

<p align="center">
  <strong>Live Avatar</strong> achieves high-fidelity, low-latency, and infinite-duration avatar generation.<br>
  <em>Compatible with NVIDIA RTX 4090 & H800 GPUs.</em>
</p>

</div>

---

## 📰 News
- **[2025/11]** Code and inference scripts released.
<!-- - **[2025/09]** Paper accepted to **CVPR/ICCV 2025**. -->

---

## 📖 Abstract

<!-- Framework Architecture Image -->
<div align="center">
  <img src="./assets/method1_00.jpg" width="95%" alt="Framework Architecture">
</div>

> **Abstract:** *Recent studies have demonstrated the effectiveness of directly aligning diffusion models... [Insert your abstract here]. In this work, we propose Live Avatar, a framework that enables streaming real-time infinite inference...*

---

## 🛠️ Installation

Please follow the steps below to set up the environment.

### 1. Create Environment
```bash
conda create -n liveavatar python=3.10 -y
conda activate liveavatar
```

### 2. Install CUDA Dependencies
```bash
conda install nvidia/label/cuda-12.4.1::cuda -y
conda install -c nvidia/label/cuda-12.4.1 cudatoolkit -y
```

### 3. Install PyTorch & Flash Attention
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install flash-attn==2.8.3 --no-build-isolation
```

### 4. Install Python Requirements
```bash
pip install -r requirements.txt
```

---

## 📥 Download Models

Please download the pretrained checkpoints from [Hugging Face](https://huggingface.co/) and place them in the `checkpoints/` directory.

| Model Component | Description | Link |
| :--- | :--- | :---: |
| `live_avatar` | our model| [Download](#) |
```bash
mkdir -p checkpoints
# Move your downloaded files here
```

---

## 🚀 Inference

### Streaming Real-time Infinite Inference


```bash
# Recommended: Run with relative path
bash infinite_inference.sh
```

---


---

## 📝 Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{yourname2025liveavatar,
  title={Live Avatar: Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference},
  author={Author One and Author Two and Author Three},
  journal={arXiv preprint arXiv:2509.XXXXX},
  year={2025}
}
```

## 🙏 Acknowledgements

We would like to express our gratitude to the following projects:

*   [CausVid](https://github.com/tianweiy/CausVid)
*   [SelfForcing](https://github.com/guandeh17/Self-Forcing)
*   [WanS2V](https://humanaigc.github.io/wan-s2v-webpage/)