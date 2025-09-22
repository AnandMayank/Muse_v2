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
