



# TITAN-Guide: Taming Inference-Time AligNment for Guided Text-to-Video Diffusion Models

👉 Paper[https://arxiv.org/abs/2508.00289]

## 📄 Overview
**TITAN-Guide** introduces a novel approach for **training-free guidance** in text-to-video diffusion models. Unlike traditional methods that require heavy fine-tuning or incur high memory costs, TITAN-Guide leverages **forward gradient descent** to optimize guidance objectives during inference efficiently.

This method enables **multi-modal control** (e.g., audio, style, aesthetics) without retraining the base model, making it ideal for creative tasks such as video generation in movie production and multimedia applications.

---

## 🚀 Key Features
- **Forward Gradient Descent** for inference-time optimization without backpropagation.
- **Multiple Gradient Initialization Strategies**:
  - Random Guess
  - Score-based Guess
  - Direct Gradient Guess
- **Supports Multi-modal Guidance**:
  - Audio-to-Video Alignment
  - Style Transfer
  - Aesthetic Control
- **Memory-Efficient**: Significant reduction in GPU VRAM usage compared to prior methods.

---

## 📊 Performance Highlights
- Outperforms state-of-the-art methods (FreeDoM, MPGD, TFG) in:
  - **Fréchet Video Distance (FVD)** – Lower is better.
  - **AV-Align Score** for audio-video consistency.
- Demonstrates robustness in **high-resolution video generation** (up to 384×384).
- Maintains **visual coherence** while integrating artistic elements.

---

## 🛠️ Installation
```bash
git clone https://github.com/sony/titan-guide.git
cd titan-guide
```

---
## 💽 Usage
Please see `./scripts` for examplar scripts. 


## 📚 Citation
If you use TITAN-Guide in your research, please cite:
```
@inproceedings{titan-guide,
  title={TITAN-Guide: Taming Inference-Time AligNment for Guided Text-to-Video Diffusion Models},
  booktitle={ICCV},
  year={2025}
}
```
