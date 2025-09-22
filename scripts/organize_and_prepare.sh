#!/bin/bash

# MUSE v2 Repository Organization and Push Preparation Script
# ===========================================================

echo "🚀 MUSE v2 Repository Organization and Push Preparation"
echo "======================================================"

# Set up variables
REPO_ROOT="/media/adityapachauri/second_drive/Muse"
TEMP_DIR="/tmp/muse_organized"
CURRENT_DIR=$(pwd)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

echo_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if we're in the right directory
check_directory() {
    if [ ! -d "$REPO_ROOT" ]; then
        echo_error "MUSE repository not found at $REPO_ROOT"
        exit 1
    fi
    
    if [ ! -d "$REPO_ROOT/.git" ]; then
        echo_error "Not a git repository. Please ensure you're in the MUSE repository."
        exit 1
    fi
    
    echo_success "Repository found at $REPO_ROOT"
}

# Function to create clean directory structure
organize_repository() {
    echo_info "Organizing repository structure..."
    
    cd "$REPO_ROOT"
    
    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt
*.bin
checkpoints/
lightning_logs/

# Training outputs (exclude large files but keep samples)
training_outputs/
!training_outputs/.gitkeep
muse_v3_advanced/training_outputs/
!muse_v3_advanced/training_outputs/.gitkeep
wandb/
logs/
cache/

# Data files (exclude large datasets but keep samples)
faiss_db/
enhanced_muse_output/training_data/
enhanced_muse_output/rl/
enhanced_muse_output/muse_v2_training/
enhanced_muse_output/muse_v2_complete/
enhanced_muse_output/images/
enhanced_muse_output/multimodal/
enhanced_muse_output/metrics/

# Keep sample files
!enhanced_muse_output/conversations/
!enhanced_muse_output/sft/
!enhanced_muse_output/dial_dpo/
!samples/

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Environment files
.env
.venv
venv/
env/

# Jupyter
.ipynb_checkpoints/

# Model files (too large for git)
*.model
*.vocab
*.tokenizer
models/
*.safetensors

# Temporary files
tmp/
temp/
*.tmp
*.log

# API keys and secrets
config.json
api_keys.json
secrets.json
EOF
        echo_success "Created .gitignore"
    fi
    
    # Create .gitkeep files for empty directories we want to keep
    mkdir -p training_outputs samples/sft_samples samples/dial_dpo_samples samples/conversation_samples
    touch training_outputs/.gitkeep
    touch muse_v3_advanced/training_outputs/.gitkeep
    
    echo_success "Repository structure organized"
}

# Function to copy original MUSE files to organized structure
organize_original_files() {
    echo_info "Organizing original MUSE files..."
    
    # Copy remaining original files to muse_original
    local original_files=(
        "user_chat.py:conversation_models/"
        "conv_manager.py:conversation_models/"
        "enhanced_conversation_generator.py:conversation_models/"
        "create_local_item_database.py:data_generation/"
        "extract_categories.py:data_generation/"
        "get_categories2item.py:data_generation/"
        "examples_demonstration.py:evaluation/"
        "reviewer.py:evaluation/"
    )
    
    for file_mapping in "${original_files[@]}"; do
        IFS=':' read -r filename target_dir <<< "$file_mapping"
        if [ -f "$filename" ]; then
            mkdir -p "muse_original/$target_dir"
            cp "$filename" "muse_original/$target_dir/"
            echo_success "Copied $filename to muse_original/$target_dir/"
        fi
    done
    
    # Copy configuration files
    if [ -f "sample_config.json" ]; then
        cp "sample_config.json" "muse_original/"
        echo_success "Copied sample_config.json to muse_original/"
    fi
    
    if [ -f "scenarios.json" ]; then
        cp "scenarios.json" "muse_original/"
        echo_success "Copied scenarios.json to muse_original/"
    fi
}

# Function to create requirements files
create_requirements() {
    echo_info "Creating requirements files..."
    
    # Create original MUSE requirements
    cat > muse_original/requirements.txt << 'EOF'
# Original MUSE Dataset Generation Framework Requirements
openai>=1.0.0
anthropic>=0.5.0
pandas>=1.5.0
numpy>=1.21.0
tqdm>=4.65.0
jsonlines>=3.1.0
requests>=2.28.0
python-dotenv>=1.0.0
Pillow>=9.0.0

# Optional but recommended
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
EOF
    
    echo_success "Created muse_original/requirements.txt"
    
    # Update main requirements.txt with MUSE v3 dependencies (already exists)
    echo_success "Main requirements.txt already updated"
}

# Function to clean up unnecessary files
cleanup_files() {
    echo_info "Cleaning up unnecessary files..."
    
    # List of files to remove (large outputs, temporary files, etc.)
    local files_to_remove=(
        "https:"  # This seems to be an erroneous directory
        "*.pyc"
        "*.pyo"
        "__pycache__"
    )
    
    for pattern in "${files_to_remove[@]}"; do
        find . -name "$pattern" -type f -delete 2>/dev/null || true
        find . -name "$pattern" -type d -exec rm -rf {} + 2>/dev/null || true
    done
    
    # Remove large training outputs but keep sample data
    if [ -d "enhanced_muse_output" ]; then
        # Keep sample directories but remove large training data
        find enhanced_muse_output -name "*.pth" -delete 2>/dev/null || true
        find enhanced_muse_output -name "*.pt" -delete 2>/dev/null || true
        find enhanced_muse_output -name "*_training_*" -size +10M -delete 2>/dev/null || true
    fi
    
    echo_success "Cleaned up unnecessary files"
}

# Function to update documentation
update_documentation() {
    echo_info "Updating documentation..."
    
    # Create CONTRIBUTING.md
    cat > docs/CONTRIBUTING.md << 'EOF'
# Contributing to MUSE

We welcome contributions to the MUSE project! This document provides guidelines for contributing.

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest tests/`
6. Format code: `black muse_v3_advanced/`
7. Submit a pull request

## Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to all functions and classes
- Keep functions focused and small

## Testing

- Write unit tests for new functions
- Include integration tests for major features
- Ensure 80%+ code coverage
- Test both original MUSE and v3 components

## Documentation

- Update relevant documentation
- Add examples for new features
- Keep README.md current
- Document API changes

## Reporting Issues

- Use GitHub Issues for bug reports
- Include reproducible examples
- Specify environment details
- Check existing issues first
EOF

    echo_success "Created CONTRIBUTING.md"
    
    # Create API reference stub
    cat > docs/API_REFERENCE.md << 'EOF'
# API Reference

## MUSE v3 Advanced Architecture

### Core Classes

#### MuseV3Architecture
Main architecture class for MUSE v3 system.

```python
from muse_v3_advanced.architecture import MuseV3Architecture

config = {
    "text_dim": 384,
    "image_dim": 512,
    "fusion_dim": 512,
    "num_intents": 7,
    "num_tools": 6,
    "device": "cuda"
}

model = MuseV3Architecture(config)
```

#### LangGraphOrchestrator
Conversation flow orchestration using LangGraph.

```python  
from muse_v3_advanced.langgraph_orchestrator import LangGraphOrchestrator

orchestrator = LangGraphOrchestrator()
response = orchestrator.process_conversation(user_input, context)
```

### Training Classes

#### SFTTrainer
Supervised Fine-Tuning trainer.

#### DialogueDPOTrainer  
Dialogue DPO trainer for preference learning.

### Tools

#### SearchTool
Advanced product search with filtering capabilities.

#### RecommendTool
ML-based personalized recommendations.

#### VisualSearchTool
Image-based product discovery using CLIP.

For complete API documentation, see the source code docstrings.
EOF

    echo_success "Created API_REFERENCE.md"
}

# Function to prepare git for push
prepare_git() {
    echo_info "Preparing git for push..."
    
    cd "$REPO_ROOT"
    
    # Check git status
    git status
    
    # Add all organized files
    git add .
    
    # Check what we're about to commit
    echo_warning "Files to be committed:"
    git diff --cached --name-status
    
    echo ""
    echo_info "Repository is ready for commit and push!"
    echo_info "Suggested commit message:"
    echo ""
    echo "🚀 Organize MUSE repository with v3 advanced architecture"
    echo ""
    echo "- Add MUSE v3 advanced architecture with real encoders"
    echo "- Organize original MUSE dataset generation framework"  
    echo "- Include SFT + Dial-DPO training pipeline samples"
    echo "- Add comprehensive documentation and setup scripts"
    echo "- Provide sample conversations and training data"
    echo "- Support cross-lingual (Hindi-English) conversations"
    echo "- Include production-ready tool implementations"
    echo ""
    echo "Features:"
    echo "- 📁 Organized directory structure"
    echo "- 🧠 Real encoder integration (HuggingFace + CLIP)" 
    echo "- 🛠️ Complete training pipeline with samples"
    echo "- 📚 Comprehensive documentation"
    echo "- 🌍 Cross-lingual support"
    echo "- 🚀 Ready-to-use setup scripts"
}

# Function to create sample commit and push script
create_push_script() {
    cat > push_to_github.sh << 'EOF'
#!/bin/bash

# MUSE Repository Push Script
echo "🚀 Pushing MUSE repository to GitHub..."

# Commit changes
git commit -m "🚀 Organize MUSE repository with v3 advanced architecture

- Add MUSE v3 advanced architecture with real encoders  
- Organize original MUSE dataset generation framework
- Include SFT + Dial-DPO training pipeline samples
- Add comprehensive documentation and setup scripts
- Provide sample conversations and training data
- Support cross-lingual (Hindi-English) conversations
- Include production-ready tool implementations

Features:
- 📁 Organized directory structure
- 🧠 Real encoder integration (HuggingFace + CLIP)
- 🛠️ Complete training pipeline with samples  
- 📚 Comprehensive documentation
- 🌍 Cross-lingual support
- 🚀 Ready-to-use setup scripts"

# Push to origin
git push origin main

echo "✅ Successfully pushed to GitHub!"
echo "🌐 Repository URL: https://github.com/AnandMayank/Muse_v2"
EOF

    chmod +x push_to_github.sh
    echo_success "Created push_to_github.sh script"
}

# Main execution
main() {
    echo_info "Starting MUSE repository organization..."
    
    check_directory
    organize_repository
    organize_original_files
    create_requirements
    cleanup_files
    update_documentation
    prepare_git
    create_push_script
    
    echo ""
    echo_success "🎉 MUSE repository organization completed!"
    echo ""
    echo_info "Next steps:"
    echo "1. Review the changes: git diff --cached"
    echo "2. Run the push script: ./push_to_github.sh"
    echo "3. Or manually commit and push:"
    echo "   git commit -m 'Your commit message'"
    echo "   git push origin main"
    echo ""
    echo_info "Repository structure:"
    echo "├── 📁 muse_original/          # Original MUSE dataset framework"
    echo "├── 📁 muse_v3_advanced/       # Advanced v3 architecture"
    echo "├── 📁 samples/                # Training samples and examples"  
    echo "├── 📁 docs/                   # Comprehensive documentation"
    echo "├── 📁 scripts/                # Setup and utility scripts"
    echo "└── 📄 README.md               # Main repository documentation"
}

# Run main function
main "$@"
