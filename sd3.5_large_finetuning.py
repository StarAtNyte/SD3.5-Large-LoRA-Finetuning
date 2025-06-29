import modal
import os
import re
from pathlib import Path
from typing import Optional, List

# =============================================================================
# MODAL APP CONFIGURATION
# =============================================================================

app = modal.App("sd35-large-carpet-finetuning")

# Base image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libgl1-mesa-glx", 
        "libglib2.0-0"
    ])
    .pip_install([
        "torch==2.3.0",
        "torchvision", 
        "diffusers==0.30.0",
        "transformers==4.42.3",
        "accelerate==0.31.0",
        "datasets",
        "wandb",
        "opencv-python",
        "albumentations",
        "Pillow",
        "safetensors",
        "xformers",
        "bitsandbytes",
        "peft>=0.11.1",
        "sentencepiece",
    ])
    .run_commands([
        "pip install --upgrade diffusers transformers accelerate"
    ])
)

# Volume configuration
volume = modal.Volume.from_name("carpet-dataset-volume", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
CACHE_PATH = "/root/huggingface_cache"

# =============================================================================
# DATASET PREPARATION
# =============================================================================

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": volume},
    timeout=3600
)
def extract_and_prepare_dataset(dataset_zip_filename: str = "carpet_data.zip") -> str:
    """Extract dataset from ZIP file if not already extracted."""
    import zipfile
    
    dataset_zip_path = Path(f"/data/{dataset_zip_filename}")
    extracted_dataset_path = Path("/data/carpet_dataset")
    
    # Check if dataset is already extracted
    if extracted_dataset_path.exists() and any(extracted_dataset_path.iterdir()):
        print("Dataset already extracted. Skipping extraction.")
        return str(extracted_dataset_path)
    
    # Verify ZIP file exists
    if not dataset_zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {dataset_zip_path}")
    
    # Extract dataset
    extracted_dataset_path.mkdir(exist_ok=True)
    print(f"Extracting {dataset_zip_path} to {extracted_dataset_path}...")
    
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dataset_path)
    
    print("Extraction completed!")
    volume.commit()
    return str(extracted_dataset_path)


# =============================================================================
# DATASET CLASS
# =============================================================================

class CarpetDataset:
    """Dataset class for carpet pattern images."""
    
    def __init__(self, root_dir: str, resolution: int = 1024):
        import cv2
        import albumentations as A
        
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.image_paths = []
        
        # Find all image files (excluding base images named '0')
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        for ext in extensions:
            for path in self.root_dir.rglob(ext):
                if path.stem != '0' and path.stem.isdigit():
                    self.image_paths.append(path)
        
        print(f"Found {len(self.image_paths)} target variation images for training.")
        
        # Image transformations
        self.transform = A.Compose([
            A.Resize(resolution, resolution, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        import torch
        import cv2
        
        try:
            img_path = self.image_paths[idx]
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transformations
            transformed = self.transform(image=image)
            pixel_values = torch.from_numpy(transformed["image"]).permute(2, 0, 1)
            
            prompt = "a photo of a high quality carpet pattern, detailed textile design, intricate weaving"
            
            return {
                "pixel_values": pixel_values,
                "prompt": prompt
            }
        except Exception as e:
            print(f"Skipping corrupted image {self.image_paths[idx]}: {e}")
            return None


def collate_fn(examples):
    """Custom collate function to handle None values from corrupted images."""
    import torch
    
    # Filter out None examples
    examples = [e for e in examples if e is not None]
    if not examples:
        return None
    
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "prompt": [e["prompt"] for e in examples]
    }


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

@app.function(
    image=image,
    gpu="A100-40GB:2",
    volumes={"/data": volume, CACHE_PATH: hf_cache_volume},
    timeout=86400,  # 24 hours
    memory=80000,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def train_sd35_large_carpet_model(
    dataset_path: str,
    output_dir: str = "/data/trained_models/sd35-large-carpet-lora",
    resume_from_checkpoint: Optional[str] = None,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    num_epochs: int = 5,
    save_every_n_epochs: int = 1,
    gradient_accumulation_steps: int = 8,
    lora_rank: int = 64,
    mixed_precision: str = "bf16",
    resolution: int = 1024,
    wandb_project: str = "sd35-large-carpet-finetuning"
):
    """Train SD3.5 Large model with LoRA on carpet dataset."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    import wandb
    
    from diffusers import SD3Transformer2DModel, AutoencoderKL
    from transformers import T5EncoderModel, CLIPTextModel, AutoTokenizer
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors
    
    # Environment setup
    os.environ["HF_HOME"] = CACHE_PATH
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_SECRET", "")
    
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in secrets.")
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb"
    )
    
    # Initialize WandB
    if accelerator.is_main_process:
        if not os.environ.get("WANDB_API_KEY"):
            print("WARNING: WandB API Key not found. Disabling logging.")
            accelerator.log_with = None
        else:
            wandb.init(project=wandb_project, config=locals(), resume="allow")
    
    # Load models
    model_id = "stabilityai/stable-diffusion-3.5-large"
    print(f"Loading models from {model_id}...")
    
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16, token=hf_token
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16, token=hf_token
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, token=hf_token
    )
    text_encoder_two = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, token=hf_token
    )
    text_encoder_three = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_3", torch_dtype=torch.bfloat16, token=hf_token
    )
    
    # Load tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", token=hf_token
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", token=hf_token
    )
    tokenizer_three = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_3", token=hf_token
    )
    
    # Freeze base models
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # Enable memory efficient attention
    transformer.enable_xformers_memory_efficient_attention()
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none"
    )
    transformer = get_peft_model(transformer, lora_config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        
        lora_safetensors_path = Path(resume_from_checkpoint) / "adapter_model.safetensors"
        lora_bin_path = Path(resume_from_checkpoint) / "adapter_model.bin"
        
        if lora_safetensors_path.exists():
            print(f"Loading weights from {lora_safetensors_path}")
            lora_weights = load_safetensors(lora_safetensors_path, device="cpu")
        elif lora_bin_path.exists():
            print(f"Loading weights from {lora_bin_path}")
            lora_weights = torch.load(lora_bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Could not find adapter weights in {resume_from_checkpoint}"
            )
        
        set_peft_model_state_dict(transformer, lora_weights)
        
        # Extract epoch number from checkpoint path
        match = re.search(r"epoch-(\d+)", resume_from_checkpoint)
        if match:
            start_epoch = int(match.group(1))
    
    if accelerator.is_main_process:
        transformer.print_trainable_parameters()
    
    # Prepare dataset and dataloader
    dataset = CarpetDataset(root_dir=dataset_path, resolution=resolution)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
    
    # Prepare with accelerator
    transformer, optimizer, train_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader
    )
    
    # Move encoders to device
    vae.to(accelerator.device)
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    text_encoder_three.to(accelerator.device)
    
    def encode_prompt(prompts: List[str]):
        """Encode prompts using all three text encoders."""
        device = accelerator.device
        
        with torch.no_grad():
            # CLIP text encoder 1
            clip_1_inputs = tokenizer_one(
                prompts,
                padding="max_length",
                max_length=tokenizer_one.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            clip_1_outputs = text_encoder_one(
                clip_1_inputs, output_hidden_states=True
            )
            
            # CLIP text encoder 2
            clip_2_inputs = tokenizer_two(
                prompts,
                padding="max_length",
                max_length=tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            clip_2_outputs = text_encoder_two(
                clip_2_inputs, output_hidden_states=True
            )
            
            # T5 text encoder
            t5_inputs = tokenizer_three(
                prompts,
                padding="max_length",
                max_length=tokenizer_three.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            t5_outputs = text_encoder_three(t5_inputs)
        
        prompt_embeds = t5_outputs.last_hidden_state
        pooled_projections = torch.cat([
            clip_1_outputs.pooler_output,
            clip_2_outputs.pooler_output
        ], dim=-1)
        
        return prompt_embeds, pooled_projections
    
    # Training loop
    print(f"Starting training from epoch {start_epoch}...")
    global_step = 0
    
    for epoch in range(start_epoch, num_epochs):
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            with accelerator.accumulate(transformer):
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=torch.bfloat16)
                    ).latent_dist.sample()
                    target_latents = latents * vae.config.scaling_factor
                
                # Add noise for flow matching
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                
                # Random timesteps
                timesteps = torch.rand(bsz, device=accelerator.device)
                timesteps = timesteps.reshape(bsz, *([1] * (target_latents.dim() - 1)))
                
                # Noisy latents and target
                noisy_latents = (1 - timesteps) * noise + timesteps * target_latents
                target = target_latents - noise
                
                # Encode prompts
                prompt_embeds, pooled_projections = encode_prompt(batch["prompt"])
                
                # Forward pass
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps.view(bsz),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_projections,
                ).sample
                
                # Compute loss
                loss = F.mse_loss(
                    model_pred.float(),
                    target.float(),
                    reduction="mean"
                )
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logging
            if accelerator.sync_gradients:
                global_step += 1
                avg_loss = accelerator.gather(loss.detach()).mean()
                progress_bar.set_postfix({"loss": f"{avg_loss.item():.4f}"})
                
                if accelerator.log_with is not None:
                    accelerator.log({"train_loss": avg_loss.item()}, step=global_step)
        
        # Save checkpoint
        if (accelerator.is_main_process and 
            (epoch + 1) % save_every_n_epochs == 0):
            
            save_path = Path(f"{output_dir}/checkpoint-epoch-{epoch+1}")
            save_path.mkdir(parents=True, exist_ok=True)
            
            accelerator.unwrap_model(transformer).save_pretrained(save_path)
            print(f"Saved LoRA checkpoint to {save_path}")
            volume.commit()
    
    # Cleanup
    if accelerator.is_main_process:
        print("Training finished. All checkpoints saved.")
        volume.commit()
        if accelerator.log_with is not None and wandb.run:
            wandb.finish()


# =============================================================================
# IMAGE GENERATION
# =============================================================================

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": volume, CACHE_PATH: hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def generate_variations(
    num_variations: int = 4,
    lora_checkpoint_path: str = "/data/trained_models/sd35-large-carpet-lora/checkpoint-epoch-1",
    input_image_path: str = "/data/carpet_dataset/carpet_1/0.jpg",
    strength: float = 0.8,
) -> str:
    """Generate multiple variations of an input image using fine-tuned LoRA."""
    import torch
    from diffusers import StableDiffusion3Img2ImgPipeline
    from PIL import Image
    
    # Setup
    model_id = "stabilityai/stable-diffusion-3.5-large"
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    
    # Load pipeline
    print(f"Loading base pipeline {model_id}...")
    pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token=hf_token
    ).to("cuda")
    
    # Load and fuse LoRA weights
    print(f"Loading LoRA weights from {lora_checkpoint_path}...")
    pipeline.load_lora_weights(lora_checkpoint_path)
    pipeline.fuse_lora()
    
    # Load input image
    print(f"Loading input image from {input_image_path}...")
    input_image = Image.open(input_image_path).convert("RGB")
    
    # Generation settings
    prompt = "a photo of a high quality carpet pattern, detailed textile design, intricate weaving"
    
    # Output directory
    output_dir = Path(f"/data/generated_variations/{Path(lora_checkpoint_path).name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving generated images to {output_dir}")
    
    # Generate variations
    for i in range(num_variations):
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        print(f"Generating variation {i+1}/{num_variations} with seed {seed}...")
        
        image_out = pipeline(
            prompt=prompt,
            image=input_image,
            strength=strength,
            num_inference_steps=28,
            generator=generator
        ).images[0]
        
        output_path = output_dir / f"variation_{i+1}_seed_{seed}.png"
        image_out.save(output_path)
    
    volume.commit()
    return str(output_dir)


# =============================================================================
# LOCAL ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def train():
    """Start a new training job from scratch."""
    print("Step 1: Extracting dataset...")
    extracted_path = extract_and_prepare_dataset.remote()
    
    print("Step 2: Starting fine-tuning job...")
    train_sd35_large_carpet_model.remote(dataset_path=extracted_path)
    
    print("Training job launched. Monitor with: modal app logs")


@app.local_entrypoint()
def resume_train(checkpoint_path: str):
    """Resume training from a specific checkpoint."""
    print("Step 1: Preparing dataset...")
    extracted_path = extract_and_prepare_dataset.remote()
    
    print(f"Step 2: Resuming training from: {checkpoint_path}")
    train_sd35_large_carpet_model.remote(
        dataset_path=extracted_path,
        resume_from_checkpoint=checkpoint_path
    )
    
    print("Resumed training job launched.")


@app.local_entrypoint()
def generate(
    epoch_num: int = 1,
    input_image_path: str = "/data/carpet_dataset/carpet_1/0.jpg",
    num_variations: int = 4
):
    """Generate image variations using a trained checkpoint."""
    checkpoint_path = f"/data/trained_models/sd35-large-carpet-lora/checkpoint-epoch-{epoch_num}"
    
    print(f"Generating {num_variations} variations using checkpoint from epoch {epoch_num}...")
    
    output_dir = generate_variations.remote(
        num_variations=num_variations,
        lora_checkpoint_path=checkpoint_path,
        input_image_path=input_image_path
    )
    
    print(f"Generation complete. Check output at: {output_dir}")


if __name__ == "__main__":
    print("SD3.5 Large Carpet Fine-tuning Pipeline")
    print("Available commands:")
    print("  modal run sd35_carpet_modal.py::train")
    print("  modal run sd35_carpet_modal.py::resume_train --checkpoint-path <path>")
    print("  modal run sd35_carpet_modal.py::generate --epoch-num <num>")