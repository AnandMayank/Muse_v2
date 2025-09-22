#!/usr/bin/env python3
"""
Quick test script for MUSE v3 Advanced System
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly"""
    try:
        from architecture import MuseV3Architecture
        logger.info("✅ MuseV3Architecture import successful")
        
        from octotools_framework import OctoToolsFramework
        logger.info("✅ OctoToolsFramework import successful")
        
        from langgraph_orchestrator import LangGraphOrchestrator
        logger.info("✅ LangGraphOrchestrator import successful")
        
        from response_generator import MuseV3ResponseGenerator
        logger.info("✅ MuseV3ResponseGenerator import successful")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_basic_initialization():
    """Test basic system initialization"""
    try:
        from architecture import TextEncoder, ImageEncoder, MetadataEncoder
        
        # Test individual components first
        text_encoder = TextEncoder(hidden_size=384)
        logger.info("✅ TextEncoder initialization successful")
        
        image_encoder = ImageEncoder(hidden_size=512)  
        logger.info("✅ ImageEncoder initialization successful")
        
        metadata_encoder = MetadataEncoder(vocab_sizes={"category": 50}, hidden_size=256)
        logger.info("✅ MetadataEncoder initialization successful")
        
        return True
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    try:
        from training_pipeline import MuseDataLoader
        
        # Initialize data loader
        data_loader = MuseDataLoader("/media/adityapachauri/second_drive/Muse")
        
        # Load data
        data = data_loader.load_all_data()
        
        logger.info(f"✅ Loaded {len(data['items'])} items")
        logger.info(f"✅ Loaded {len(data['conversations'])} conversations")
        logger.info(f"✅ Loaded {len(data['user_profiles'])} user profiles")
        
        return True
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Testing MUSE v3 Advanced System")
    logger.info("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Initialization", test_basic_initialization),
        ("Data Loading", test_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! MUSE v3 Advanced system is working!")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
