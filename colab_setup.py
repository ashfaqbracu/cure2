"""
Colab Setup Script for CureBench with Unsloth

Run this script at the beginning of your Google Colab session to set up 
the environment for CureBench evaluation with Unsloth models.

Usage:
    1. Upload this script to your Colab session
    2. Run: exec(open('colab_setup.py').read())
    3. Follow the instructions to clone your repo and load models
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def setup_colab_environment():
    """Set up Google Colab environment for CureBench with Unsloth"""
    
    print("🏥 CureBench + Unsloth Colab Setup")
    print("=" * 50)
    
    # Check GPU
    print("🔍 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ GPU memory: {gpu_memory:.1f} GB")
        else:
            print("⚠️ No GPU detected. Consider switching to GPU runtime for better performance.")
    except ImportError:
        print("⚠️ PyTorch not available yet")
    
    # Install Unsloth
    print("\n📦 Installing Unsloth...")
    commands = [
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
        'pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes'
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            print("❌ Unsloth installation failed. Please check the error messages.")
            return False
    
    # Install other dependencies
    print("\n📦 Installing other dependencies...")
    other_deps = [
        'pip install transformers torch tqdm pandas',
        'pip install jupyter matplotlib seaborn'
    ]
    
    for cmd in other_deps:
        run_command(cmd, f"Installing: {cmd.split()[-1]}")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import torch
        from unsloth import FastModel
        import pandas as pd
        import tqdm
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n📋 Next Steps:")
    print("1. Clone your CureBench repository:")
    print("   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git")
    print("   %cd YOUR_REPO")
    print("")
    print("2. Load your Unsloth model:")
    print("   from unsloth import FastModel")
    print("   model, tokenizer = FastModel.from_pretrained('unsloth/medgemma-4b-it-bnb-4bit', ...)")
    print("")
    print("3. Run evaluation:")
    print("   # Use the provided Jupyter notebook or Python scripts")
    print("")
    print("🎉 Setup completed! You're ready to run CureBench with Unsloth models.")
    
    return True

# Additional helper functions for Colab
def clone_repo(repo_url, repo_name=None):
    """Helper function to clone repository"""
    if repo_name is None:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
    
    print(f"📥 Cloning repository: {repo_url}")
    if run_command(f"git clone {repo_url}", "Repository cloning"):
        print(f"✅ Repository cloned to: {repo_name}")
        return repo_name
    return None

def download_dataset(dataset_url, filename):
    """Helper function to download dataset"""
    print(f"📥 Downloading dataset: {filename}")
    if run_command(f"wget -O {filename} {dataset_url}", "Dataset download"):
        print(f"✅ Dataset downloaded: {filename}")
        return True
    return False

def show_memory_usage():
    """Show current memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🔥 GPU Memory: {allocated:.1f}GB used, {reserved:.1f}GB reserved, {total:.1f}GB total")
        else:
            print("ℹ️ No GPU available")
    except ImportError:
        print("ℹ️ PyTorch not available")

# Run setup if script is executed directly
if __name__ == "__main__":
    setup_colab_environment()
else:
    print("🔧 CureBench Colab setup module loaded")
    print("Run setup_colab_environment() to set up your environment")
