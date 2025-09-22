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
