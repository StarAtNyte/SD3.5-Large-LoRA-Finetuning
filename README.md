---
base_model: stabilityai/stable-diffusion-3.5-large
library_name: peft
---

# SD3.5-Large Carpet Diffusion: Image to Variations

This repository contains a Gradio application for generating variations of an input image using the Stable Diffusion 3.5 Large model, finetuned for carpet patterns. It leverages `mmgp` for optimized memory management, enabling efficient inference even on systems with limited VRAM.

## Model Details

### Model Description

This model is a finetuned version of `stabilityai/stable-diffusion-3.5-large` specifically adapted for generating high-quality carpet patterns. It uses a LoRA (Low-Rank Adaptation) checkpoint to apply the finetuning. The application allows users to upload an image and generate 5 unique variations based on a text prompt and various parameters like strength, inference steps, and guidance scale.

- **Developed by:** StarAtNyte
- **Model type:** Image-to-Image Diffusion Model (Finetuned)
- **Language(s) (NLP):** English
- **License:** MIT (or appropriate license for SD3.5-Large and finetuning)
- **Finetuned from model:** `stabilityai/stable-diffusion-3.5-large`

### Model Sources

- **Repository:** [This GitHub Repository]
- **Demo:** Local Gradio application (`app_optimized.py`)

## Uses

### Direct Use

This model is intended for generating creative variations of carpet patterns from an input image. It can be used by designers, artists, or anyone interested in exploring generative AI for textile design.

### Out-of-Scope Use

This model is not intended for generating harmful, illegal, or unethical content. It is specifically trained for carpet patterns and may not perform well on other image generation tasks without further finetuning.

## Bias, Risks, and Limitations

As with any generative AI model, there may be inherent biases present in the training data which could be reflected in the generated outputs. Users should be aware that the model's output is influenced by the input prompt and image, and results may vary.

### Recommendations

Users should critically evaluate the generated images and understand that the model is a tool for creative exploration.

## How to Get Started with the Model

### Prerequisites

- Python 3.8+
- Git

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPO_URL_HERE]
    cd Carpet Diffusion
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Model Weights:**
    The `adapter_model.safetensors` and `adapter_config.json` files are required for the application to run. These are typically finetuned model weights. You will need to place these files in the root directory of the project.
    *(Note: These files are `.gitignore`d and not part of the repository due to their size and nature as model artifacts.)*

### Running the Application

To launch the Gradio application, run:

```bash
python app_optimized.py
```

The application will typically be accessible in your browser at `http://192.168.50.145:7860` (or a similar local IP address/port).

## Training Details

### Training Data

The model was finetuned on a dataset of carpet patterns. (Specific details about the dataset are not provided in the current context but would typically be included here).

### Training Procedure

The model was finetuned using LoRA (Low-Rank Adaptation) on the `stabilityai/stable-diffusion-3.5-large` base model. The `sd3.5_large_finetuning.py` script likely contains the finetuning logic.

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision (inferred from `app_optimized.py` using `torch.bfloat16`)

## Evaluation

(More information needed for detailed evaluation metrics and results.)

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** (e.g., NVIDIA RTX 4090, inferred from `app_optimized.py` comments)
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications

### Model Architecture and Objective

The application uses `StableDiffusion3Img2ImgPipeline` from the `diffusers` library, optimized with `mmgp` for efficient memory management during inference. The objective is to generate diverse image variations from an input image and text prompt.

### Compute Infrastructure

#### Hardware

Optimized for GPUs with sufficient VRAM (e.g., RTX 4090 with 24GB VRAM) and system RAM (e.g., 32GB+). `mmgp` profiles are available for various hardware configurations.

#### Software

- Python
- PyTorch
- Gradio
- Diffusers
- PIL (Pillow)
- mmgp

## Model Card Authors

StarAtNyte

## Model Card Contact

khanalnitij20@gmail.com

### Framework versions

- PEFT 0.15.2
