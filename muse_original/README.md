# Original MUSE: Dataset Generation Framework

This directory contains the original MUSE dataset generation framework as described in the paper "MUSE: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles".

## Overview

The original MUSE framework focuses on generating high-quality conversational recommendation datasets with:

- **Scenario-grounded user profiles**: Realistic user personas based on demographics and preferences
- **Multi-turn conversations**: Natural dialogue flows for product discovery
- **Multimodal interactions**: Text-based conversations with product image references
- **Quality evaluation**: Comprehensive metrics for conversation quality assessment

## Framework Components

### 1. Data Generation Pipeline

**Key Files:**
- `generate_user_profiles.py`: Creates diverse user personas with demographics and preferences
- `setup_conversation_generation.py`: Initializes conversation generation system
- `generate_convs.py`: Main conversation generation script
- `create_local_item_database.py`: Sets up local product database for recommendations

**User Profile Generation:**
```python
# Example user profile generation
python generate_user_profiles.py \
    --num_profiles 1000 \
    --demographics_file user_demographics.json \
    --output_file user_profiles_output.json
```

**Conversation Generation:**
```python  
# Generate conversations from user profiles
python generate_convs.py \
    --user_profiles user_profiles_output.json \
    --num_conversations 5000 \
    --output_dir generated_conversations/
```

### 2. Conversation Models

**Components:**
- `system_chat.py`: System-side conversation handler
- `user_chat.py`: User simulation for realistic interactions  
- `conv_manager.py`: Manages conversation flow and state
- `enhanced_conversation_generator.py`: Advanced conversation generation with quality controls

**Features:**
- Multi-turn dialogue simulation
- Context-aware response generation
- Product recommendation integration
- Natural conversation flow management

### 3. User Profiling System

**User Scenario Generation:**
- Demographic-based profile creation
- Interest and preference modeling
- Behavioral pattern simulation
- Cultural and regional adaptations

**Profile Structure:**
```json
{
  "user_id": "user_12345",
  "demographics": {
    "age": 28,
    "gender": "female", 
    "location": "Mumbai",
    "income_level": "middle"
  },
  "preferences": {
    "style": "casual",
    "budget_range": "1000-5000",
    "preferred_brands": ["Zara", "H&M"],
    "occasions": ["office", "casual"]
  },
  "conversation_style": {
    "language": "mixed",  // Hindi-English mix
    "formality": "casual",
    "detail_preference": "moderate"
  }
}
```

### 4. Evaluation Framework

**Quality Metrics:**
- Conversation coherence scoring
- Response relevance evaluation  
- Multi-turn consistency checking
- User satisfaction simulation

**Evaluation Scripts:**
```python
# Evaluate generated conversations
python evaluation/evaluate_conversations.py \
    --conversation_file generated_conversations.json \
    --metrics coherence,relevance,satisfaction
```

## Dataset Statistics

The original MUSE dataset contains:

- **User Profiles**: 10,000+ diverse user personas
- **Conversations**: 50,000+ multi-turn dialogues
- **Products**: 100,000+ product entries with metadata
- **Languages**: Hindi, English, and mixed (Hinglish)
- **Domains**: Fashion, electronics, home & garden, books

## Usage Examples

### Quick Start

```bash
# 1. Set up the environment
pip install -r requirements.txt

# 2. Generate user profiles
python generate_user_profiles.py --num_profiles 100

# 3. Set up conversation system
python setup_conversation_generation.py

# 4. Generate sample conversations  
python generate_convs.py --num_conversations 500

# 5. Evaluate conversation quality
python evaluation/evaluate_conversations.py
```

### Custom Dataset Generation

```python
from conversation_models.system_chat import SystemChat
from user_profiling.generate_user_profiles import UserProfileGenerator
from data_generation.generate_convs import ConversationGenerator

# Create custom user profiles
profile_gen = UserProfileGenerator()
profiles = profile_gen.generate_profiles(
    num_profiles=1000,
    demographics_config="custom_demographics.json"
)

# Generate conversations with custom settings
conv_gen = ConversationGenerator()
conversations = conv_gen.generate_conversations(
    user_profiles=profiles,
    conversation_length_range=(3, 10),
    product_categories=["fashion", "electronics"]
)

# Save results
conv_gen.save_conversations(conversations, "custom_dataset.json")
```

### Integration with MUSE v3

The original MUSE framework provides the foundational dataset that can be used to train the advanced MUSE v3 architecture:

```python
# Load original MUSE data for v3 training
from muse_v3_advanced.training_scripts.data_generation_pipeline import MuseDataLoader

# Load original conversations for training data
data_loader = MuseDataLoader()
training_data = data_loader.load_original_muse_conversations(
    conversation_file="generated_conversations.json",
    user_profiles="user_profiles.json"
)

# Convert to v3 training format
v3_training_data = data_loader.convert_to_toolformer_format(training_data)
```

## Configuration

### Environment Setup

```bash
# Core dependencies for original MUSE
pip install openai>=1.0.0
pip install anthropic>=0.5.0  
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install tqdm>=4.65.0
pip install jsonlines>=3.1.0
```

### API Configuration

Create a `config.json` file:

```json
{
  "openai_api_key": "your_openai_key_here",
  "anthropic_api_key": "your_anthropic_key_here", 
  "rate_limits": {
    "requests_per_minute": 60,
    "max_retries": 3
  },
  "conversation_settings": {
    "max_turns": 10,
    "temperature": 0.7,
    "response_timeout": 30
  }
}
```

## Data Quality & Evaluation

### Quality Metrics

1. **Coherence Score**: Measures logical flow across conversation turns
2. **Relevance Score**: Evaluates response appropriateness to user queries  
3. **Diversity Score**: Assesses variety in generated conversations
4. **Naturalness Score**: Rates human-like conversation quality

### Sample Quality Report

```
Dataset Quality Report
=====================
Total Conversations: 10,000
Average Turns per Conversation: 6.4
Language Distribution:
  - English: 35%
  - Hindi: 25% 
  - Mixed (Hinglish): 40%

Quality Scores (1-5 scale):
  - Coherence: 4.2 ± 0.6
  - Relevance: 4.1 ± 0.7
  - Diversity: 3.9 ± 0.5
  - Naturalness: 4.0 ± 0.6
```

## Files Overview

### Core Generation Scripts
- `generate_user_profiles.py`: User persona generation
- `generate_convs.py`: Main conversation generation
- `setup_conversation_generation.py`: System initialization
- `create_local_item_database.py`: Product database setup

### Conversation Management  
- `system_chat.py`: System response generation
- `user_chat.py`: User behavior simulation
- `conv_manager.py`: Conversation flow control
- `enhanced_conversation_generator.py`: Quality-enhanced generation

### Data Processing
- `extract_categories.py`: Product category extraction
- `get_categories2item.py`: Category-item mapping
- `prepare_sft_data.py`: SFT training data preparation
- `prepare_dpo_data.py`: DPO training data preparation

### Evaluation & Analysis
- `examples_demonstration.py`: Sample conversation examples
- `reviewer.py`: Automated conversation quality review
- Various analysis and metrics scripts

This original framework provides the foundation for generating the high-quality conversational datasets that power the advanced MUSE v3 architecture.
