# Script to upload your model to Hugging Face Hub
# Run this once to upload your model

from huggingface_hub import HfApi, create_repo
import torch

# Your Hugging Face username and model name
HF_USERNAME = "your-username"  # Replace with your HF username
MODEL_NAME = "potato-disease-detection"

# Create repository on Hugging Face
repo_id = f"{HF_USERNAME}/{MODEL_NAME}"

try:
    create_repo(repo_id, exist_ok=True)
    print(f"Repository created: {repo_id}")
    
    # Upload model file
    api = HfApi()
    api.upload_file(
        path_or_fileobj="potato_disease_model.pth",
        path_in_repo="potato_disease_model.pth",
        repo_id=repo_id,
    )
    print("Model uploaded successfully!")
    print(f"Your model URL: https://huggingface.co/{repo_id}")
    
except Exception as e:
    print(f"Error: {e}")
    print("Please install huggingface_hub: pip install huggingface_hub")
    print("And login: huggingface-cli login")