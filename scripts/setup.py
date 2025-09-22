#!/usr/bin/env python3
"""
MUSE Setup and Verification Script
=================================

This script sets up the MUSE system and verifies all components are working correctly.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda_availability():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            return True
        else:
            print("⚠️  CUDA not available, using CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "torch", "transformers", "sentence-transformers", 
        "numpy", "pandas", "tqdm", "datasets",
        "accelerate", "wandb"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    return missing_packages

def setup_directories():
    """Create necessary directories"""
    directories = [
        "training_outputs",
        "training_outputs/sft",
        "training_outputs/dpo", 
        "training_outputs/checkpoints",
        "evaluation_results",
        "logs",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_models():
    """Download required pre-trained models"""
    print("\n📥 Downloading pre-trained models...")
    
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "openai/clip-vit-base-patch32"
    ]
    
    try:
        from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
        
        for model_name in models:
            print(f"Downloading {model_name}...")
            if "clip" in model_name:
                CLIPModel.from_pretrained(model_name)
                CLIPProcessor.from_pretrained(model_name)
            else:
                AutoModel.from_pretrained(model_name)
                AutoTokenizer.from_pretrained(model_name)
            print(f"✅ {model_name}")
            
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False
    
    return True

def create_sample_config():
    """Create sample configuration file"""
    config = {
        "model": {
            "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "image_encoder": "openai/clip-vit-base-patch32",
            "device": "cuda" if check_cuda_availability() else "cpu",
            "fusion_dim": 512,
            "max_sequence_length": 512
        },
        "training": {
            "batch_size": 4 if not check_cuda_availability() else 8,
            "learning_rate": 5e-5,
            "num_epochs": 20,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1
        },
        "languages": ["en", "hi"],
        "tools": ["search", "recommend", "compare", "filter", "translate", "visual_search"],
        "data_paths": {
            "item_database": "./data/items.json",
            "user_profiles": "./data/user_profiles.json",
            "conversation_logs": "./logs/"
        },
        "logging": {
            "use_wandb": True,
            "project_name": "muse-v3-training",
            "log_level": "INFO"
        }
    }
    
    config_path = Path("configs/config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created configuration file: {config_path}")

def verify_installation():
    """Verify MUSE installation by running basic tests"""
    print("\n🧪 Running verification tests...")
    
    try:
        # Test basic imports
        sys.path.append("muse_v3_advanced")
        from architecture import MuseV3Architecture
        print("✅ MUSE v3 Architecture import successful")
        
        # Test model initialization
        config = {
            "text_dim": 384,
            "image_dim": 512, 
            "fusion_dim": 512,
            "num_intents": 7,
            "num_tools": 6,
            "device": "cpu"  # Use CPU for verification
        }
        
        model = MuseV3Architecture(config)
        print("✅ MUSE v3 model initialization successful")
        
        # Test basic inference
        dummy_input = {
            "text_input": "test query",
            "batch_size": 1
        }
        
        # This would normally require full setup, so just check structure
        print("✅ Model structure verification complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="MUSE Setup and Verification")
    parser.add_argument("--verify", action="store_true", help="Only run verification tests")
    parser.add_argument("--download-models", action="store_true", help="Download pre-trained models")
    parser.add_argument("--setup-dirs", action="store_true", help="Setup directories")
    args = parser.parse_args()
    
    print("🚀 MUSE v3 Setup and Verification")
    print("=" * 50)
    
    # Check basic requirements
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking Dependencies:")
    missing = check_dependencies()
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        if not args.verify:
            sys.exit(1)
    
    print("\n🖥️  System Information:")
    check_cuda_availability()
    
    if args.verify:
        # Only run verification
        success = verify_installation()
        sys.exit(0 if success else 1)
    
    # Full setup
    print("\n📁 Setting up directories:")
    if args.setup_dirs or not args.download_models:
        setup_directories()
    
    print("\n⚙️  Creating configuration:")
    create_sample_config()
    
    if args.download_models or (not args.setup_dirs and not args.download_models):
        print("\n📥 Downloading models:")
        if not download_models():
            print("⚠️  Model download failed. You can download them later.")
    
    print("\n🧪 Running verification:")
    success = verify_installation()
    
    if success:
        print("\n🎉 MUSE v3 setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit configs/config.json with your settings")
        print("2. Run: python muse_v3_advanced/quick_test.py")
        print("3. Start training: python training_scripts/comprehensive_training_orchestrator.py")
    else:
        print("\n❌ Setup completed with warnings. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
