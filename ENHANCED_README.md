# MUSe v2: Enhanced Agentic Conversation System

This is the enhanced version of the MUSe (Multimodal Conversational Recommendation) system, implementing advanced improvements for training modern agentic conversation models including DiaToolDPO, VisTA, ToolFormer, and similar approaches.

## 🚀 Key Improvements

### 1. Data Quality & Realism
- **Metadata Normalization**: Automatic cleaning of item data, duplicate removal, category standardization
- **Natural Dialogue Variations**: Injected hesitations, corrections, clarifications for human-like flow
- **Failure Recovery**: Realistic failure scenarios with proper recovery turns
- **Multimodal Grounding**: Every recommendation properly linked to item metadata and images

### 2. Richer Context
- **Synthetic User Personas**: Age, style preferences, budget, purchase history
- **Contextual Dialogue Goals**: Specific objectives for evaluation (e.g., "Find red dress under $50")
- **Intent Labeling**: Every turn tagged with intent (search, compare, filter, recommend, etc.)
- **Conversation Continuity**: Pre-loaded user history and preference tracking

### 3. Tool-Ready Architecture
- **Tool Call Integration**: Recommendations converted to explicit tool calls
- **Alternative Generation**: Multiple tool options for SFT & DPO training
- **Success Rate Tracking**: Realistic tool performance simulation
- **Multi-Strategy Support**: Content-based, collaborative, hybrid, persona-based recommendations

## 📁 System Architecture

```
enhanced_system/
├── tool_manager.py           # Tool selection and execution
├── persona_system.py         # User persona generation
├── enhanced_conversation_manager.py  # Main conversation orchestration
├── enhanced_user_chat.py     # Improved user simulation
├── enhanced_system_chat.py   # Advanced recommendation system
└── config.py                 # System configuration
```

## 🛠 Installation & Setup

```bash
# Install dependencies
pip install openai pydantic tqdm

# Configure the system
cp enhanced_system/config.py config.py
# Edit config.py with your API keys and paths

# Run the enhanced system
python enhanced_main.py
```

## 📊 Training Data Pipeline

The sys
### Stage 1: SFT (Supervised Fine-Tuning)
```json
{
  "context": ["User: Show me something in blue under $100"],
  "tool_calls": [{"tool": "search_items", "args": {"color":"blue","max_price":100}}],
  "response": "Here are some great blue options...",
  "intent": "recommend"
}
```

### Stage 2: DPO (Direct Preference Optimization)
```json
{tem generates three types of training data:

  "context": "Conversation history...",
  "chosen": {
    "tool_calls": [{"tool": "search_items", "args": {"color":"blue","max_price":100}}],
    "response": "Perfect blue dress within budget",
    "success_rate": 0.95
  },
  "rejected": {
    "tool_calls": [{"tool": "filter_items", "args": {"price_range": [0,1000]}}],
    "response": "Here are some options",
    "success_rate": 0.60
  }
}
```

### Stage 3: RL (Reinforcement Learning)
```json
{
  "state": "Current conversation context",
  "action": {"tool": "recommend_similar", "args": {"item_id": "123"}},
  "reward": 0.85,
  "next_state": "Updated conversation context"
}
```

## 🎯 Key Features

### Enhanced User Personas
- **Demographics**: Age range, occupation, lifestyle
- **Preferences**: Style, budget, colors, brands, sizes
- **Context**: Purchase history, wardrobe gaps, upcoming events
- **Personality**: Traits affecting shopping behavior

### Advanced Tool System
- **8 Core Tools**: Search, fetch details, compare, filter, recommend similar, check availability, get reviews, visual search
- **Smart Selection**: LLM-powered tool selection with context awareness
- **Alternative Generation**: Multiple tool options for training
- **Failure Simulation**: Realistic error scenarios and recovery

### Conversation Quality
- **Natural Variations**: Hesitations, corrections, clarifications
- **Multimodal Integration**: Image references and visual search
- **Failure Recovery**: Graceful error handling with alternative approaches
- **Intent Tracking**: Comprehensive intent labeling for analysis

## 📈 Evaluation Metrics

The system tracks comprehensive metrics:

- **Conversation Success Rate**: Percentage of successful completions
- **Tool Usage Frequency**: How often tools are effectively used
- **Intent Distribution**: Balance of different conversation intents
- **Vocabulary Diversity**: Richness of generated language
- **Persona Alignment**: How well recommendations match user profiles
- **Multimodal Grounding**: Percentage of turns with proper image references

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Enable/disable features
ENABLE_TOOLS = True
ENABLE_MULTIMODAL = True
ENABLE_PERSONAS = True
ENABLE_FAILURE_SIMULATION = True

# Conversation parameters
MIN_TURNS = 4
MAX_TURNS = 12
CHIT_CHAT_PROBABILITY = 0.25
FAILURE_RATE = 0.15

# Training data generation
GENERATE_SFT_DATA = True
GENERATE_DPO_DATA = True
GENERATE_RL_DATA = True
```

## 🚀 Usage Examples

### Basic Conversation Generation
```python
from enhanced_main import EnhancedMuseSystem

# Initialize system
system = EnhancedMuseSystem(
    api_base="your_api_base",
    api_key="your_api_key",
    db_path="path_to_database",
    data_path="path_to_data",
    model_name="path_to_model"
)

# Generate conversations
system.generate_enhanced_conversations(user_scenarios, num_conversations=1000)
```

### Tool Integration Example
```python
from enhanced_system.tool_manager import ToolManager, ToolType

# Use tools directly
tool_call = ToolCall(
    tool_type=ToolType.SEARCH_ITEMS,
    args={"query": "blue dress", "max_price": 100}
)

result = tool_manager.execute_tool(tool_call)
```

### Persona Generation
```python
from enhanced_system.persona_system import PersonaGenerator

generator = PersonaGenerator()
persona = generator.generate_persona()

print(f"Generated persona: {persona.age_range.value}, {persona.style_preferences}")
```

## 📊 Output Structure

The system generates organized output:

```
enhanced_muse_output/
├── sft/
│   └── sft_training_data.json
├── dpo/
│   └── dpo_training_data.json
├── rl/
│   └── rl_training_data.json
├── conversations/
│   ├── conversation_001.json
│   └── ...
├── metrics/
│   └── conversation_metrics.json
├── dataset_statistics.json
├── generation_report.json
└── validation_report.json
```

## 🎓 Training Pipeline

### For SFT (Supervised Fine-Tuning):
1. Use `sft_training_data.json` with explicit tool annotations
2. Train model to predict correct tool usage
3. Focus on context → tool selection mapping

### For DPO (Direct Preference Optimization):
1. Use `dpo_training_data.json` with chosen/rejected pairs
2. Train model to prefer better tool selections
3. Optimize for success rate and user satisfaction

### For RL (Reinforcement Learning):
1. Use `rl_training_data.json` as episode data
2. Simulate conversation environment
3. Reward successful recommendations and efficient tool usage

## 🔍 Quality Assurance

The system includes comprehensive quality checks:

- **Data Completeness**: Ensures all required fields are present
- **Conversation Quality**: Validates natural flow and coherence
- **Tool Usage Analysis**: Checks realistic tool usage patterns
- **Persona Diversity**: Ensures varied user representations
- **Dialogue Naturalness**: Measures human-likeness of conversations

## 🤝 Integration with Existing Tools

The enhanced system is designed to work with:

- **DiaToolDPO**: Direct preference optimization for tool usage
- **VisTA**: Visual-text alignment in multimodal conversations
- **ToolFormer**: Tool-augmented language model training
- **OctoTools**: Multi-tool selection and usage
- **Other tool-based conversation systems**

## 📝 Citation

If you use this enhanced system in your research, please cite:

```bibtex
@software{muse_v2_enhanced,
  title={MUSe v2: Enhanced Agentic Conversation System with Tool Integration},
  author={Enhanced MUSe Team},
  year={2024},
  url={https://github.com/your-repo/enhanced-muse}
}
```

## 🛣 Roadmap

Future improvements planned:

- [ ] Integration with more advanced multimodal models
- [ ] Support for real-time conversation evaluation
- [ ] Advanced persona learning from conversation history
- [ ] Integration with external APIs and databases
- [ ] Support for multiple languages and cultural contexts
- [ ] Advanced tool chaining and composite actions

## 📧 Support

For questions and support, please open an issue or contact the development team.

---

**Note**: This enhanced system represents a significant advancement over the original MUSe, specifically designed for training next-generation agentic conversation models with proper tool integration and realistic user simulation.
