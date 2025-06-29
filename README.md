# SD3.5 Large Carpet Pattern Fine-tuning
A comprehensive Modal-based pipeline for fine-tuning Stable Diffusion 3.5 Large on carpet pattern datasets using LoRA (Low-Rank Adaptation).

![image](https://github.com/user-attachments/assets/fea74698-5046-4028-8707-64942f4eb73f)
![image](https://github.com/user-attachments/assets/7f55ed5a-48f7-4ca4-a387-e8943acb8df9)

## 🎯 Overview

This project enables you to:
- Fine-tune SD3.5 Large on custom carpet pattern datasets
- Generate high-quality carpet pattern variations
- Resume training from checkpoints
- Monitor training with WandB integration
- Scale efficiently using Modal's cloud infrastructure

## 🏗️ Architecture

- **Base Model**: Stable Diffusion 3.5 Large
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Cloud Platform**: Modal Labs
- **GPU Requirements**: A100-40GB (2x for training, 1x for inference)
- **Monitoring**: Weights & Biases (WandB)

## 🚀 Quick Start

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Hugging Face Token**: Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **WandB Account** (optional): For training monitoring

### Installation

```bash
# Install Modal CLI
pip install modal

# Login to Modal
modal auth new

# Clone this repository
git clone <your-repo-url>
cd sd35-carpet-finetuning
```

### Setup Secrets

Configure your secrets in Modal:

```bash
# Hugging Face token (required)
modal secret create huggingface-secret HUGGINGFACE_TOKEN=your_hf_token_here

# WandB API key (optional, for training monitoring)
modal secret create wandb-secret WANDB_API_KEY=your_wandb_key_here
```

### Dataset Preparation

1. **Dataset Structure**: Organize your carpet images like this:
   ```
   carpet_data.zip
   ├── carpet_1/
   │   ├── 0.jpg          # Base image (will be ignored in training)
   │   ├── 1.jpg          # Variation 1
   │   ├── 2.jpg          # Variation 2
   │   └── ...
   ├── carpet_2/
   │   ├── 0.jpg
   │   ├── 1.jpg
   │   └── ...
   └── ...
   ```

2. **Upload Dataset**: Upload `carpet_data.zip` to your Modal volume:
   ```bash
   modal volume put carpet-dataset-volume carpet_data.zip /carpet_data.zip
   ```

## 🎮 Usage

### Training Commands

#### Start New Training
```bash
modal run sd35_carpet_modal.py::train
```

#### Resume from Checkpoint
```bash
modal run sd35_carpet_modal.py::resume_train --checkpoint-path "/data/trained_models/sd35-large-carpet-lora/checkpoint-epoch-3"
```

#### Generate Variations
```bash
modal run sd35_carpet_modal.py::generate --epoch-num 5 --num-variations 8
```

### Advanced Configuration

You can customize training parameters by modifying the function calls:

```python
# Custom training parameters
train_sd35_large_carpet_model.remote(
    dataset_path=extracted_path,
    learning_rate=2e-5,
    batch_size=2,
    num_epochs=10,
    lora_rank=128,
    resolution=1024,
    gradient_accumulation_steps=4
)
```

## ⚙️ Configuration Parameters

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-5` | Learning rate for AdamW optimizer |
| `batch_size` | `1` | Batch size per GPU |
| `num_epochs` | `5` | Number of training epochs |
| `lora_rank` | `64` | LoRA rank (higher = more parameters) |
| `gradient_accumulation_steps` | `8` | Steps to accumulate gradients |
| `resolution` | `1024` | Training image resolution |
| `mixed_precision` | `"bf16"` | Mixed precision training |
| `save_every_n_epochs` | `1` | Checkpoint saving frequency |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_variations` | `4` | Number of variations to generate |
| `strength` | `0.8` | Img2img strength (0.0-1.0) |
| `num_inference_steps` | `28` | Denoising steps |

## 📁 File Structure

```
sd35-carpet-finetuning/
├── sd35_carpet_modal.py     # Main Modal application
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## 🔍 Monitoring

### WandB Integration

If you've configured WandB secrets, training metrics will be automatically logged:
- Training loss
- Learning rate
- Epoch progress
- System metrics

Access your runs at: `https://wandb.ai/your-username/sd35-large-carpet-finetuning`

### Modal Logs

Monitor your training in real-time:
```bash
# List running apps
modal app list

# View logs for specific app
modal app logs sd35-large-carpet-finetuning
```

## 💾 Data Management

### Volumes

The pipeline uses two Modal volumes:
- `carpet-dataset-volume`: Stores datasets and model checkpoints
- `huggingface-cache`: Caches Hugging Face models

### Checkpoint Structure

```
/data/trained_models/sd35-large-carpet-lora/
├── checkpoint-epoch-1/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── checkpoint-epoch-2/
└── ...
```

### Generated Images

```
/data/generated_variations/checkpoint-epoch-X/
├── variation_1_seed_12345.png
├── variation_2_seed_67890.png
└── ...
```


### Performance Tips

2. **LoRA Rank**: Higher ranks (128, 256) may give better results but use more memory
3. **Resolution**: 1024px is optimal, but 512px trains faster
4. **Gradient Accumulation**: Increase to simulate larger batch sizes

## 📊 Expected Results

### Training Time
- **5 epochs**: ~2-4 hours (depending on dataset size)
- **Checkpoint size**: ~130MB per checkpoint
- **GPU utilization**: ~80-90% on A100-40GB

### Quality Metrics
- Loss typically decreases from ~0.1 to ~0.01
- Generated images show improved carpet pattern fidelity after epoch 3-5
- Best results usually achieved between epochs 5-10

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Stability AI](https://stability.ai/) for Stable Diffusion 3.5
- [Hugging Face](https://huggingface.co/) for the Diffusers library
- [Modal Labs](https://modal.com/) for cloud infrastructure
- [Microsoft](https://github.com/microsoft/LoRA) for LoRA technique


