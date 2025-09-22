# Installation Guide

## Prerequisites

### Hardware Requirements

**Minimum Requirements:**
- **GPU**: NVIDIA RTX 3080 or equivalent (10GB VRAM) 
- **RAM**: 16GB system memory
- **Storage**: 100GB available space for models and data
- **CPU**: 8-core modern processor (Intel i7/AMD Ryzen 7 or better)

**Recommended for Production:**
- **GPU**: NVIDIA A100, V100, or RTX 4090 (24GB+ VRAM)
- **RAM**: 64GB system memory
- **Storage**: 1TB NVMe SSD
- **CPU**: 16-core Xeon or equivalent
- **Network**: High-bandwidth for real-time inference

### Software Prerequisites

- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **Git**: Latest version with Git LFS support
- **Docker**: Optional, for containerized deployment

## Installation Steps

### 1. Clone the Repository

```bash
# Clone with Git LFS support for large model files
git lfs install
git clone https://github.com/AnandMayank/Muse_v2.git
cd Muse_v2
```

### 2. Set Up Python Environment

**Option A: Using Conda (Recommended)**
```bash
# Create conda environment
conda create -n muse_v3 python=3.9
conda activate muse_v3

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

**Option B: Using pip + venv**
```bash
# Create virtual environment
python -m venv muse_v3_env
source muse_v3_env/bin/activate  # On Windows: muse_v3_env\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Install MUSE v3 Advanced Components

```bash
# Navigate to MUSE v3 directory
cd muse_v3_advanced

# Install v3-specific dependencies
pip install -r requirements.txt

# Install MUSE v3 in development mode
pip install -e .
```

### 4. Download Pre-trained Models

```bash
# Download required models automatically
python scripts/download_models.py

# Or manually specify models to download
python scripts/download_models.py --models clip,sentence-transformers,bert-base
```

The script will download:
- **CLIP Model**: `openai/clip-vit-base-patch32` (~600MB)
- **Text Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (~90MB) 
- **Intent Classifier**: Custom trained model (~200MB)
- **Dialogue State Tracker**: LSTM-based model (~50MB)

### 5. Configuration Setup

```bash
# Copy sample configuration
cp configs/sample_config.json configs/config.json

# Edit configuration file
nano configs/config.json  # Or use your preferred editor
```

**Sample Configuration:**
```json
{
  "model": {
    "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "image_encoder": "openai/clip-vit-base-patch32",
    "device": "cuda:0",
    "fusion_dim": 512,
    "max_sequence_length": 512
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "num_epochs": 20,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1
  },
  "languages": ["en", "hi"],
  "tools": ["search", "recommend", "compare", "filter", "translate", "visual_search"],
  "data_paths": {
    "item_database": "/path/to/item/database",
    "user_profiles": "/path/to/user/profiles",
    "conversation_logs": "/path/to/conversation/logs"
  },
  "api_keys": {
    "openai_api_key": "your_openai_key_here",
    "huggingface_token": "your_hf_token_here"
  }
}
```

### 6. Verify Installation

```bash
# Run system verification
python scripts/setup.py --verify

# Test core components
python muse_v3_advanced/test_system.py

# Quick functionality test
python muse_v3_advanced/quick_test.py
```

Expected output:
```
✅ MUSE v3 Installation Verification
✅ PyTorch: 2.1.0 (CUDA available: True)
✅ Transformers: 4.35.0
✅ CLIP Model: Loaded successfully
✅ Text Encoder: Loaded successfully  
✅ LangGraph: 0.1.0
✅ All dependencies satisfied
🚀 MUSE v3 ready for use!
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config.json
"batch_size": 4  # Instead of 8

# Enable gradient checkpointing
"gradient_checkpointing": true

# Use mixed precision training
"use_amp": true
```

**2. Model Download Issues**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/storage

# Use HuggingFace token for private models
export HUGGINGFACE_HUB_TOKEN=your_token_here

# Manual model download
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
"
```

**3. Import Errors**
```bash
# Ensure MUSE v3 is in Python path
export PYTHONPATH=$PYTHONPATH:/path/to/Muse_v2/muse_v3_advanced

# Or add to your script:
import sys
sys.path.append('/path/to/Muse_v2/muse_v3_advanced')
```

**4. Permission Issues (Linux/macOS)**
```bash
# Fix permission for model cache
sudo chown -R $USER:$USER ~/.cache/huggingface

# Make scripts executable
chmod +x scripts/*.py
```

### Platform-Specific Notes

**Windows:**
- Use Windows Subsystem for Linux (WSL2) for better compatibility
- Install Visual Studio Build Tools for C++ compilation
- Use PowerShell or Git Bash for commands

**macOS:**
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Metal Performance Shaders for acceleration on Apple Silicon
- Set `device: "mps"` in config for M1/M2 Macs

**Linux:**
- Ensure NVIDIA drivers and CUDA toolkit are properly installed
- Check CUDA compatibility: `nvidia-smi`
- Use system package manager for dependencies when possible

## Development Setup

### For Contributing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ --cov=muse_v3_advanced

# Format code
black muse_v3_advanced/
isort muse_v3_advanced/
```

### IDE Configuration

**VS Code:**
- Install Python extension
- Configure Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
- Add to settings.json:
```json
{
    "python.defaultInterpreterPath": "./muse_v3_env/bin/python",
    "python.terminal.activateEnvironment": true
}
```

**PyCharm:**
- Configure Project Interpreter: File → Settings → Project → Python Interpreter
- Add MUSE v3 directory to Python path
- Enable scientific mode for Jupyter notebook support

## Docker Installation (Alternative)

```bash
# Build Docker image
docker build -t muse_v3 .

# Run with GPU support
docker run --gpus all -it -v $(pwd):/workspace muse_v3

# Run specific component
docker run --gpus all muse_v3 python muse_v3_advanced/quick_test.py
```

## Next Steps

After installation:

1. **Run Demo**: `python muse_v3_advanced/conversation_demo.py`
2. **Training**: Follow [Training Guide](TRAINING.md)
3. **API Setup**: See [Deployment Guide](DEPLOYMENT.md) 
4. **Development**: Check [Contributing Guidelines](CONTRIBUTING.md)

## Getting Help

- **GitHub Issues**: [Report installation problems](https://github.com/AnandMayank/Muse_v2/issues)
- **Discussions**: [Ask questions](https://github.com/AnandMayank/Muse_v2/discussions)
- **Documentation**: [Wiki pages](https://github.com/AnandMayank/Muse_v2/wiki)
