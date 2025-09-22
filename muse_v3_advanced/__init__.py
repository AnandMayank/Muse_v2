"""
MUSE v3 Advanced Multimodal Conversational AI
==============================================

A comprehensive system featuring:
- Perception Layers (Text, Image, Metadata Encoders)
- Tool-Oriented Policy Layer (OctoTools-inspired)
- Cross-lingual and Bilingual Support
- Multimodal Fusion and Reasoning
- Advanced Training Pipeline (SFT + DPO + RL)
"""

__version__ = "3.0.0"
__author__ = "MUSE Team"
__description__ = "Advanced Multimodal Conversational AI with Tool-Oriented Architecture"

# Core components
from .perception.text_encoder import TextEncoder
from .perception.image_encoder import ImageEncoder
from .perception.metadata_encoder import MetadataEncoder
from .perception.multimodal_fusion import MultimodalFusion

from .dialogue.state_tracker import DialogueStateTracker
from .dialogue.intent_classifier import IntentClassifier

from .tools.tool_selector import ToolSelector
from .tools.argument_generator import ArgumentGenerator
from .tools.tool_planner import ToolPlanner
from .tools.cross_lingual_tools import CrossLingualTools

from .response.response_generator import ResponseGenerator
from .response.bilingual_templates import BilingualTemplates

from .training.sft_trainer import SFTTrainer
from .training.dpo_trainer import DPOTrainer
from .training.rl_trainer import RLTrainer
from .training.multimodal_trainer import MultimodalTrainer

from .core.muse_v3_system import MUSEv3System

__all__ = [
    'TextEncoder', 'ImageEncoder', 'MetadataEncoder', 'MultimodalFusion',
    'DialogueStateTracker', 'IntentClassifier',
    'ToolSelector', 'ArgumentGenerator', 'ToolPlanner', 'CrossLingualTools',
    'ResponseGenerator', 'BilingualTemplates',
    'SFTTrainer', 'DPOTrainer', 'RLTrainer', 'MultimodalTrainer',
    'MUSEv3System'
]
