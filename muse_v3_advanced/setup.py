#!/usr/bin/env python3
"""
MUSE v3 Advanced System Setup
=============================

Setup script for MUSE v3 system initialization and configuration.
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create necessary directory structure"""
    logger.info("Creating directory structure...")
    
    base_dir = Path(__file__).parent
    directories = [
        "models",
        "logs", 
        "cache",
        "data",
        "configs",
        "tests"
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def create_sample_config():
    """Create sample configuration file"""
    logger.info("Creating sample configuration...")
    
    config = {
        "system": {
            "name": "MUSE v3 Advanced",
            "version": "3.0.0",
            "environment": "development",
            "debug_mode": True
        },
        "data": {
            "base_path": "/media/adityapachauri/second_drive/Muse",
            "models_dir": "./models",
            "logs_dir": "./logs",
            "cache_dir": "./cache"
        },
        "architecture": {
            "text_dim": 768,
            "image_dim": 512,
            "metadata_dim": 256,
            "fusion_dim": 512,
            "hidden_dim": 256,
            "num_intents": 7,
            "state_dim": 128,
            "num_tools": 6,
            "device": "cuda"
        },
        "languages": {
            "supported": ["en", "hi"],
            "default": "en",
            "auto_detect": True,
            "translation_enabled": True
        },
        "tools": {
            "search_enabled": True,
            "recommend_enabled": True,
            "compare_enabled": True,
            "filter_enabled": True,
            "translate_enabled": True,
            "visual_search_enabled": True,
            "max_results": 10,
            "timeout": 30.0
        }
    }
    
    config_path = Path(__file__).parent / "configs" / "muse_v3_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample configuration created: {config_path}")

def check_dependencies():
    """Check if required dependencies are available"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        "torch",
        "transformers", 
        "numpy",
        "pandas",
        "scikit-learn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are available!")
    return True

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "muse_v3_setup.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.info(f"Logging setup complete. Log file: {log_file}")

def main():
    """Main setup function"""
    logger.info("Starting MUSE v3 Advanced System Setup...")
    
    try:
        # Setup logging
        setup_logging()
        
        # Create directory structure
        create_directory_structure()
        
        # Create sample configuration
        create_sample_config()
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Setup incomplete due to missing dependencies")
            sys.exit(1)
        
        logger.info("MUSE v3 Advanced System setup complete!")
        logger.info("Next steps:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Configure system: edit configs/muse_v3_config.json")
        logger.info("3. Run system: python main_system.py")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
