# -*- coding: utf-8 -*-
"""
Configuration file for Enhanced MUSe System
Update the values below with your actual API keys and paths
"""

import os
from pathlib import Path

# =============================================================================
# API Configuration
# =============================================================================

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')  # Using smaller, cheaper model

# Alternative API Configuration (if using different provider)
API_BASE = os.getenv('API_BASE', OPENAI_BASE_URL)
API_KEY = os.getenv('API_KEY', OPENAI_API_KEY)

# =============================================================================
# Database and Model Configuration  
# =============================================================================

# Local database path for item storage
DB_PATH = os.getenv('DB_PATH', './item_database.db')

# Path to item profile JSON file
DATA_PATH = os.getenv('DATA_PATH', './updated_item_profile.json')

# Embedding model path (for BGE-M3 or similar)
MODEL_PATH = os.getenv('MODEL_PATH', 'BAAI/bge-m3')

# =============================================================================
# System Configuration
# =============================================================================

# Output directories
OUTPUT_DIR = './enhanced_muse_output'
CONVERSATION_DIR = f'{OUTPUT_DIR}/conversations'
TRAINING_DIR = f'{OUTPUT_DIR}/training_data'
LOGS_DIR = './logs'

# Image storage directory
IMAGES_DIR = './images_main'

# =============================================================================
# API Rate Limiting Configuration
# =============================================================================

# Rate limiting settings to avoid quota issues
REQUESTS_PER_MINUTE = 2  # Very conservative - only 2 requests per minute
DELAY_BETWEEN_REQUESTS = 30  # Wait 30 seconds between API calls
MAX_TOKENS_PER_REQUEST = 100  # Reduce token usage even more
TEMPERATURE = 0.3  # Lower temperature for more consistent responses

# =============================================================================
# Generation Settings
# =============================================================================

# Conversation generation settings (reduced to avoid quota issues)
MAX_CONVERSATIONS = 50  # Reduced from 1000
MAX_TURNS_PER_CONVERSATION = 6  # Reduced from 12
MIN_TURNS_PER_CONVERSATION = 3  # Reduced from 6

# Feature toggles
ENABLE_TOOLS = True
ENABLE_MULTIMODAL = True
ENABLE_PERSONAS = True
ENABLE_FAILURE_SIMULATION = True

# Failure simulation rates
SYSTEM_FAILURE_RATE = 0.05
TOOL_FAILURE_RATE = 0.10
GENERAL_FAILURE_RATE = 0.15

# =============================================================================
# Training Data Configuration
# =============================================================================

# Training data export settings
SAVE_SFT_DATA = True
SAVE_DPO_DATA = True
SAVE_RL_DATA = True
SAVE_METRICS = True

# Data quality thresholds
MIN_CONVERSATION_LENGTH = 4
MIN_SUCCESS_RATE = 0.3
MIN_VOCABULARY_DIVERSITY = 50

# =============================================================================
# Utility Functions
# =============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        OUTPUT_DIR,
        CONVERSATION_DIR,
        TRAINING_DIR,
        LOGS_DIR,
        IMAGES_DIR,
        f'{TRAINING_DIR}/sft',
        f'{TRAINING_DIR}/dpo', 
        f'{TRAINING_DIR}/rl',
        f'{TRAINING_DIR}/metrics'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return directories

def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Check API configuration
    if API_KEY in ['your-openai-api-key-here', 'key', 'api_key']:
        issues.append("❌ API_KEY not configured properly")
    
    if API_BASE in ['base_url', 'api_base']:
        issues.append("❌ API_BASE not configured properly")
    
    # Check file paths
    if not Path(DATA_PATH).exists() and DATA_PATH != './updated_item_profile.json':
        issues.append(f"❌ DATA_PATH file not found: {DATA_PATH}")
    
    # Check directories
    try:
        ensure_directories()
    except Exception as e:
        issues.append(f"❌ Cannot create directories: {e}")
    
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ Configuration validated successfully")
        return True

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        "api_configured": API_KEY not in ['your-openai-api-key-here', 'key', 'api_key'],
        "data_path_exists": Path(DATA_PATH).exists(),
        "output_dir": OUTPUT_DIR,
        "features": {
            "tools": ENABLE_TOOLS,
            "multimodal": ENABLE_MULTIMODAL,
            "personas": ENABLE_PERSONAS,
            "failure_simulation": ENABLE_FAILURE_SIMULATION
        },
        "generation_limits": {
            "max_conversations": MAX_CONVERSATIONS,
            "max_turns": MAX_TURNS_PER_CONVERSATION
        }
    }

if __name__ == "__main__":
    print("🔧 Enhanced MUSe Configuration")
    print("=" * 40)
    
    # Validate configuration
    is_valid = validate_config()
    
    # Show configuration summary
    summary = get_config_summary()
    print(f"\n📊 Configuration Summary:")
    print(f"API Configured: {'✅' if summary['api_configured'] else '❌'}")
    print(f"Data Path Exists: {'✅' if summary['data_path_exists'] else '❌'}")
    print(f"Output Directory: {summary['output_dir']}")
    
    print(f"\n🎯 Enabled Features:")
    for feature, enabled in summary['features'].items():
        print(f"  {feature}: {'✅' if enabled else '❌'}")
    
    if not is_valid:
        print(f"\n🛠 To fix configuration issues:")
        print(f"1. Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print(f"2. Or edit this file directly and replace placeholder values")
        print(f"3. Ensure required data files exist")
        print(f"4. Run: python config.py to validate again")
