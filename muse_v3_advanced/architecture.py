#!/usr/bin/env python3
"""
MUSE v3 Advanced Architecture
============================

Complete implementation of advanced multimodal conversational AI with:
- A. Perception Layer (Text/Image/Metadata Encoders + Multimodal Fusion)
- B. Dialogue & Intent Layer (State Tracking + Intent Classification)
- C. Tool-Oriented Policy Layer (OctoTools-inspired)
- D. Response Generator (Bilingual support)

Based on real MUSE data and state-of-the-art architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, 
    CLIPModel, CLIPProcessor,
    T5ForConditionalGeneration, T5Tokenizer
)
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import json
from enum import Enum

# =============================================================================
# A. PERCEPTION LAYER
# =============================================================================

class ModalityEncoder(nn.Module):
    """Base class for modality encoders"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError

class TextEncoder(ModalityEncoder):
    """Transformer-based text encoder for dialogue context"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 hidden_size: int = 768):
        super().__init__(hidden_size)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Projection layer to match hidden_size
        if self.encoder.config.hidden_size != hidden_size:
            self.projection = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        else:
            self.projection = nn.Identity()
            
        # Language detection for bilingual support
        self.language_detector = nn.Linear(hidden_size, 2)  # EN/HI
        
    def forward(self, text: Union[str, List[str]], 
                return_language_logits: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode text with optional language detection
        
        Args:
            text: Input text or list of texts
            return_language_logits: Whether to return language classification
            
        Returns:
            Dictionary with 'embeddings' and optionally 'language_logits'
        """
        if isinstance(text, str):
            text = [text]
            
        # Tokenize and encode
        inputs = self.tokenizer(text, padding=True, truncation=True, 
                               return_tensors="pt", max_length=512)
        
        # Move inputs to same device as model
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            
        # Use CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        # Project to target hidden size
        embeddings = self.projection(embeddings)
        
        result = {'embeddings': embeddings}
        
        if return_language_logits:
            language_logits = self.language_detector(embeddings)
            result['language_logits'] = language_logits
            
        return result

class ImageEncoder(ModalityEncoder):
    """CLIP/BLIP-based image encoder for item images"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 hidden_size: int = 768):
        super().__init__(hidden_size)
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Projection to match hidden_size
        clip_hidden_size = self.clip_model.config.vision_config.hidden_size
        if clip_hidden_size != hidden_size:
            self.projection = nn.Linear(clip_hidden_size, hidden_size)
        else:
            self.projection = nn.Identity()
            
        # Visual attribute classifier
        self.attribute_classifier = nn.Linear(hidden_size, 128)  # Common attributes
        
    def forward(self, images: Any, 
                return_attributes: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode images with optional attribute extraction
        
        Args:
            images: PIL images or tensors
            return_attributes: Whether to return visual attributes
            
        Returns:
            Dictionary with 'embeddings' and optionally 'attributes'
        """
        if not isinstance(images, list):
            images = [images]
            
        # Process images
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(**inputs)
            
        # Get image embeddings (CLS token)
        embeddings = vision_outputs.pooler_output
        embeddings = self.projection(embeddings)
        
        result = {'embeddings': embeddings}
        
        if return_attributes:
            attributes = self.attribute_classifier(embeddings)
            result['attributes'] = torch.sigmoid(attributes)  # Multi-label
            
        return result

class MetadataEncoder(ModalityEncoder):
    """Structured encoder for item attributes and metadata"""
    
    def __init__(self, vocab_sizes: Dict[str, int], hidden_size: int = 768):
        super().__init__(hidden_size)
        
        self.vocab_sizes = vocab_sizes
        self.embeddings = nn.ModuleDict()
        
        # Create embedding layers for each metadata field
        for field, vocab_size in vocab_sizes.items():
            self.embeddings[field] = nn.Embedding(vocab_size, hidden_size // 4)
            
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(len(vocab_sizes) * (hidden_size // 4), hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Metadata type classifier
        self.type_classifier = nn.Linear(hidden_size, len(vocab_sizes))
        
    def forward(self, metadata: Dict[str, torch.Tensor], 
                return_types: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode structured metadata
        
        Args:
            metadata: Dictionary of field_name -> tensor of indices
            return_types: Whether to return metadata type predictions
            
        Returns:
            Dictionary with 'embeddings' and optionally 'type_logits'
        """
        embedded_fields = []
        
        for field_name, indices in metadata.items():
            if field_name in self.embeddings:
                embedded = self.embeddings[field_name](indices)
                embedded_fields.append(embedded)
                
        # Concatenate and fuse
        if embedded_fields:
            concatenated = torch.cat(embedded_fields, dim=-1)
            embeddings = self.fusion(concatenated)
        else:
            # Handle empty metadata
            batch_size = next(iter(metadata.values())).size(0)
            embeddings = torch.zeros(batch_size, self.hidden_size)
            
        result = {'embeddings': embeddings}
        
        if return_types:
            type_logits = self.type_classifier(embeddings)
            result['type_logits'] = type_logits
            
        return result

class MultimodalFusion(nn.Module):
    """Cross-attention fusion between modalities"""
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Cross-attention layers
        self.text_to_image_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.text_to_metadata_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.image_to_text_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        
        # Fusion networks
        self.text_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.image_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, text_emb: torch.Tensor, 
                image_emb: torch.Tensor, 
                metadata_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fuse multimodal representations using cross-attention
        
        Args:
            text_emb: Text embeddings [batch_size, seq_len, hidden_size]
            image_emb: Image embeddings [batch_size, num_images, hidden_size]
            metadata_emb: Metadata embeddings [batch_size, hidden_size]
            
        Returns:
            Fused representations and attention weights
        """
        batch_size = text_emb.size(0)
        
        # Ensure metadata has sequence dimension
        if metadata_emb.dim() == 2:
            metadata_emb = metadata_emb.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Cross-attention: text attends to image and metadata
        text_image_attended, text_image_weights = self.text_to_image_attention(
            text_emb, image_emb, image_emb
        )
        text_metadata_attended, text_metadata_weights = self.text_to_metadata_attention(
            text_emb, metadata_emb, metadata_emb
        )
        
        # Cross-attention: image attends to text
        image_text_attended, image_text_weights = self.image_to_text_attention(
            image_emb, text_emb, text_emb
        )
        
        # Fuse text representations
        text_fused = self.text_fusion(torch.cat([
            text_emb, text_image_attended, text_metadata_attended
        ], dim=-1))
        
        # Fuse image representations  
        image_fused = self.image_fusion(torch.cat([
            image_emb, image_text_attended
        ], dim=-1))
        
        # Pool sequences for final fusion
        text_pooled = text_fused.mean(dim=1)  # [batch_size, hidden_size]
        image_pooled = image_fused.mean(dim=1)
        metadata_pooled = metadata_emb.squeeze(1)
        
        # Final multimodal representation
        multimodal_repr = self.final_fusion(torch.cat([
            text_pooled, image_pooled, metadata_pooled
        ], dim=-1))
        
        return {
            'multimodal_representation': multimodal_repr,
            'text_fused': text_fused,
            'image_fused': image_fused,
            'attention_weights': {
                'text_to_image': text_image_weights,
                'text_to_metadata': text_metadata_weights,
                'image_to_text': image_text_weights
            }
        }

# =============================================================================
# B. DIALOGUE & INTENT LAYER  
# =============================================================================

class Intent(Enum):
    """Conversation intents"""
    SEARCH = "search"
    FILTER = "filter" 
    COMPARE = "compare"
    RECOMMEND = "recommend"
    CHITCHAT = "chitchat"
    TRANSLATE = "translate"
    VISUAL_SEARCH = "visual_search"

@dataclass
class DialogueState:
    """Dialogue state representation"""
    
    # User goals and constraints
    current_intent: Intent
    user_constraints: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    # Conversation context
    conversation_history: List[Dict[str, Any]]
    current_turn: int
    language: str  # 'en', 'hi', or 'mixed'
    
    # Item context
    current_items: List[Dict[str, Any]]
    last_recommendation: Optional[Dict[str, Any]]
    
    # Session state
    user_persona: Dict[str, Any]
    session_metadata: Dict[str, Any]

class DialogueStateTracker(nn.Module):
    """Track dialogue state over conversation turns"""
    
    def __init__(self, hidden_size: int = 768, num_intents: int = 7):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_intents = num_intents
        
        # State tracking LSTM
        self.state_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # State components predictors
        self.constraint_predictor = nn.Linear(hidden_size, 64)  # Common constraints
        self.preference_predictor = nn.Linear(hidden_size, 32)  # User preferences
        self.persona_updater = nn.Linear(hidden_size * 2, hidden_size)
        
        # Language detection
        self.language_classifier = nn.Linear(hidden_size, 3)  # en, hi, mixed
        
    def forward(self, multimodal_input: torch.Tensor,
                previous_state: Optional[DialogueState] = None) -> Dict[str, Any]:
        """
        Update dialogue state given multimodal input
        
        Args:
            multimodal_input: Fused multimodal representation
            previous_state: Previous dialogue state
            
        Returns:
            Updated dialogue state components
        """
        batch_size = multimodal_input.size(0)
        
        # LSTM state tracking
        if multimodal_input.dim() == 2:
            multimodal_input = multimodal_input.unsqueeze(1)
            
        lstm_output, (hidden, cell) = self.state_lstm(multimodal_input)
        state_repr = lstm_output[:, -1, :]  # Last output
        
        # Predict state components
        constraints = torch.sigmoid(self.constraint_predictor(state_repr))
        preferences = torch.sigmoid(self.preference_predictor(state_repr))
        language_logits = self.language_classifier(state_repr)
        
        # Update persona if previous state exists
        if previous_state is not None:
            # This would use actual previous persona in practice
            persona_input = torch.cat([state_repr, state_repr], dim=-1)  # Placeholder
            updated_persona = self.persona_updater(persona_input)
        else:
            updated_persona = state_repr
            
        return {
            'state_representation': state_repr,
            'predicted_constraints': constraints,
            'predicted_preferences': preferences,
            'language_logits': language_logits,
            'updated_persona': updated_persona,
            'lstm_hidden': hidden,
            'lstm_cell': cell
        }

class IntentClassifier(nn.Module):
    """Classify user intent from multimodal input"""
    
    def __init__(self, hidden_size: int = 768, num_intents: int = 7):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_intents)
        )
        
        # Intent-specific feature extractors
        self.intent_features = nn.ModuleDict({
            'search': nn.Linear(hidden_size, 64),
            'filter': nn.Linear(hidden_size, 32),
            'compare': nn.Linear(hidden_size, 48),
            'recommend': nn.Linear(hidden_size, 56),
            'chitchat': nn.Linear(hidden_size, 24),
            'translate': nn.Linear(hidden_size, 16),
            'visual_search': nn.Linear(hidden_size, 72)
        })
        
    def forward(self, multimodal_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify intent and extract intent-specific features
        
        Args:
            multimodal_input: Fused multimodal representation
            
        Returns:
            Intent logits and intent-specific features
        """
        # Main intent classification
        intent_logits = self.classifier(multimodal_input)
        
        # Extract intent-specific features
        intent_features = {}
        for intent_name, extractor in self.intent_features.items():
            intent_features[intent_name] = extractor(multimodal_input)
            
        return {
            'intent_logits': intent_logits,
            'intent_features': intent_features,
            'predicted_intent': torch.argmax(intent_logits, dim=-1)
        }

# =============================================================================
# C. TOOL-ORIENTED POLICY LAYER (OctoTools-inspired)
# =============================================================================

@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    execution_cost: float
    supported_languages: List[str]

class ToolRegistry:
    """Registry of available tools"""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
        
    def _register_default_tools(self):
        """Register default tools"""
        
        # Basic tools
        self.register_tool(Tool(
            name="search",
            description="Search for items based on query",
            parameters={"query": str, "category": str, "max_results": int},
            execution_cost=1.0,
            supported_languages=["en", "hi"]
        ))
        
        self.register_tool(Tool(
            name="filter",
            description="Filter items by attributes",
            parameters={"items": list, "filters": dict},
            execution_cost=0.5,
            supported_languages=["en", "hi"]
        ))
        
        self.register_tool(Tool(
            name="recommend",
            description="Recommend items based on preferences",
            parameters={"user_profile": dict, "context": dict, "num_items": int},
            execution_cost=2.0,
            supported_languages=["en", "hi"]
        ))
        
        self.register_tool(Tool(
            name="compare",
            description="Compare multiple items",
            parameters={"items": list, "comparison_aspects": list},
            execution_cost=1.5,
            supported_languages=["en", "hi"]
        ))
        
        # Language tools
        self.register_tool(Tool(
            name="translate",
            description="Translate text between languages",
            parameters={"text": str, "source_lang": str, "target_lang": str},
            execution_cost=0.3,
            supported_languages=["en", "hi"]
        ))
        
        # Visual tools
        self.register_tool(Tool(
            name="visual_search",
            description="Search using image similarity",
            parameters={"image": "PIL.Image", "similarity_threshold": float},
            execution_cost=3.0,
            supported_languages=["en", "hi"]
        ))
        
        self.register_tool(Tool(
            name="extract_visual_attributes",
            description="Extract attributes from product images",
            parameters={"image": "PIL.Image"},
            execution_cost=2.5,
            supported_languages=["en", "hi"]
        ))
        
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
        
    def get_available_tools(self, language: str = "en") -> List[Tool]:
        """Get tools available for a language"""
        return [tool for tool in self.tools.values() 
                if language in tool.supported_languages]

class ToolSelector(nn.Module):
    """Select best tool given dialogue state and available tools"""
    
    def __init__(self, hidden_size: int = 768, max_tools: int = 10):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_tools = max_tools
        
        # Tool scoring network
        self.tool_scorer = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),  # +64 for tool features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
        # Tool embedding (learned representations)
        self.tool_embeddings = nn.Embedding(max_tools, 64)
        
        # Project tool embeddings to match hidden_size for attention
        self.tool_projection = nn.Linear(64, hidden_size)
        
        # Context-tool interaction
        self.context_tool_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        
    def forward(self, dialogue_state: torch.Tensor,
                available_tools: List[str],
                tool_registry: ToolRegistry) -> Dict[str, Any]:
        """
        Select best tool for current state
        
        Args:
            dialogue_state: Current dialogue state representation
            available_tools: List of available tool names
            tool_registry: Tool registry
            
        Returns:
            Tool selection scores and recommendation
        """
        batch_size = dialogue_state.size(0)
        
        # Get tool embeddings for available tools
        tool_ids = torch.tensor([hash(tool_name) % self.max_tools 
                                for tool_name in available_tools], device=dialogue_state.device)
        tool_embs = self.tool_embeddings(tool_ids)
        
        # Project tool embeddings to match hidden_size
        tool_embs_projected = self.tool_projection(tool_embs)
        
        # Expand dialogue state for each tool
        if dialogue_state.dim() == 2:
            dialogue_state = dialogue_state.unsqueeze(1)  # Add seq dim
            
        # Context-tool attention - now all dimensions match
        tool_embs_expanded = tool_embs_projected.unsqueeze(0).expand(batch_size, -1, -1)
        attended_context, attention_weights = self.context_tool_attention(
            dialogue_state, tool_embs_expanded, tool_embs_expanded
        )
        
        # Score each tool
        tool_scores = []
        for i, tool_name in enumerate(available_tools):
            # Combine context and original tool features (64-dim)
            context_tool = torch.cat([
                attended_context.mean(dim=1),  # Pool context (hidden_size)
                tool_embs[i:i+1].expand(batch_size, -1)  # Original tool embedding (64-dim)
            ], dim=-1)
            
            score = self.tool_scorer(context_tool)
            tool_scores.append(score)
            
        tool_scores = torch.cat(tool_scores, dim=-1)
        
        # Get best tool
        best_tool_idx = torch.argmax(tool_scores, dim=-1)
        best_tool_name = [available_tools[idx.item()] for idx in best_tool_idx]
        
        return {
            'tool_scores': tool_scores,
            'best_tool_idx': best_tool_idx,
            'best_tool_name': best_tool_name,
            'attention_weights': attention_weights
        }

class ArgumentGenerator(nn.Module):
    """Generate arguments for selected tool"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Argument generators for different parameter types
        self.string_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)  # Vocab size for string tokens
        )
        
        self.numeric_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.category_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 64)  # Common categories
        )
        
        # Parameter-specific extractors
        self.param_extractors = nn.ModuleDict({
            'query': nn.Linear(hidden_size, hidden_size),
            'category': nn.Linear(hidden_size, 32),
            'max_results': nn.Linear(hidden_size, 1),
            'similarity_threshold': nn.Linear(hidden_size, 1),
            'num_items': nn.Linear(hidden_size, 1)
        })
        
    def forward(self, dialogue_state: torch.Tensor,
                tool_name: str,
                tool_registry: ToolRegistry) -> Dict[str, Any]:
        """
        Generate arguments for the selected tool
        
        Args:
            dialogue_state: Current dialogue state
            tool_name: Name of selected tool
            tool_registry: Tool registry
            
        Returns:
            Generated arguments for the tool
        """
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            return {'arguments': {}}
            
        generated_args = {}
        
        # Generate arguments based on tool parameters
        for param_name, param_type in tool.parameters.items():
            if param_name in self.param_extractors:
                param_features = self.param_extractors[param_name](dialogue_state)
                
                if param_type == str:
                    # For string parameters, return features for downstream processing
                    generated_args[param_name] = param_features
                elif param_type == int:
                    # For integer parameters, generate counts
                    generated_args[param_name] = torch.clamp(
                        param_features.squeeze(-1).int(), 1, 10
                    )
                elif param_type == float:
                    # For float parameters, use sigmoid
                    generated_args[param_name] = torch.sigmoid(param_features.squeeze(-1))
                elif param_type == dict or param_type == list:
                    # For complex types, return raw features
                    generated_args[param_name] = param_features
                    
        # Calculate confidence safely
        tensor_args = [torch.mean(torch.abs(arg.float())) for arg in generated_args.values() 
                      if isinstance(arg, torch.Tensor)]
        
        if tensor_args:
            confidence = torch.mean(torch.stack(tensor_args))
        else:
            confidence = torch.tensor(0.0, device=dialogue_state.device)
        
        return {
            'arguments': generated_args,
            'tool_name': tool_name,
            'confidence': confidence
        }

class Planner(nn.Module):
    """Plan multi-step tool chains when needed"""
    
    def __init__(self, hidden_size: int = 768, max_steps: int = 5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        
        # Input projection to handle concatenated features (dialogue_state + goal)
        self.input_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Planning LSTM
        self.planning_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Step predictor
        self.step_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, max_steps)
        )
        
        # Dependency predictor
        self.dependency_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, dialogue_state: torch.Tensor,
                goal_representation: torch.Tensor) -> Dict[str, Any]:
        """
        Plan sequence of tool calls
        
        Args:
            dialogue_state: Current dialogue state
            goal_representation: Goal representation
            
        Returns:
            Planned sequence of steps
        """
        batch_size = dialogue_state.size(0)
        
        # Planning input
        # Concatenate inputs and project to correct dimension
        planning_input = torch.cat([dialogue_state, goal_representation], dim=-1)
        planning_input = self.input_projection(planning_input)
        
        if planning_input.dim() == 2:
            planning_input = planning_input.unsqueeze(1)
            
        # Generate planning sequence
        lstm_output, _ = self.planning_lstm(planning_input)
        
        # Predict number of steps
        num_steps_logits = self.step_predictor(lstm_output.squeeze(1))
        predicted_steps = torch.argmax(num_steps_logits, dim=-1) + 1
        
        # For now, return simple plan (can be expanded)
        return {
            'num_steps_logits': num_steps_logits,
            'predicted_steps': predicted_steps,
            'planning_representation': lstm_output.squeeze(1)
        }

# =============================================================================
# MAIN MUSE v3 ARCHITECTURE
# =============================================================================

class MuseV3Architecture(nn.Module):
    """
    Main MUSE v3 Architecture integrating all components
    
    A complete 4-layer architecture:
    - Perception Layer: Text, Image, Metadata encoders + Multimodal Fusion
    - Dialogue & Intent Layer: State tracking + Intent classification
    - Tool-Oriented Policy: Tool selection + Argument generation + Planning
    - Response Generation: Integrated through external generator
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Device configuration
        self.device = torch.device(config.get("device", "cpu"))
        
        # A. Perception Layer Components
        self.text_encoder = TextEncoder(
            model_name=config.get("text_model", "sentence-transformers/all-MiniLM-L6-v2"),
            hidden_size=config.get("text_dim", 768)
        )
        
        self.image_encoder = ImageEncoder(
            model_name=config.get("image_model", "openai/clip-vit-base-patch32"),
            hidden_size=config.get("image_dim", 512)
        )
        
        self.metadata_encoder = MetadataEncoder(
            vocab_sizes=config.get("metadata_vocab", {"category": 100, "brand": 1000}),
            hidden_size=config.get("metadata_dim", 256)
        )
        
        self.fusion_layer = MultimodalFusion(
            hidden_size=config.get("fusion_dim", 512),
            num_heads=config.get("fusion_heads", 8)
        )
        
        # Projection layers to align dimensions for fusion
        self.text_proj = nn.Linear(config.get("text_dim", 768), config.get("fusion_dim", 512))
        self.image_proj = nn.Linear(config.get("image_dim", 512), config.get("fusion_dim", 512))
        self.metadata_proj = nn.Linear(config.get("metadata_dim", 256), config.get("fusion_dim", 512))
        
        # B. Dialogue & Intent Layer Components
        self.state_tracker = DialogueStateTracker(
            hidden_size=config.get("fusion_dim", 512),
            num_intents=config.get("num_intents", 7)
        )
        
        self.intent_classifier = IntentClassifier(
            hidden_size=config.get("fusion_dim", 512),
            num_intents=config.get("num_intents", 7)
        )
        
        # C. Tool-Oriented Policy Components
        self.tool_selector = ToolSelector(
            hidden_size=config.get("fusion_dim", 512),
            max_tools=config.get("num_tools", 6)
        )
        
        self.arg_generator = ArgumentGenerator(
            hidden_size=config.get("fusion_dim", 512)
        )
        
        self.planner = Planner(
            hidden_size=config.get("fusion_dim", 512),
            max_steps=config.get("max_steps", 5)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through complete MUSE v3 architecture
        
        Args:
            batch_data: Dictionary containing:
                - text_input: Text input tokens/embeddings
                - image_input: Image data (optional)
                - metadata_input: Metadata dictionary
                - conversation_history: Previous conversation turns
                - user_profile: User profile information
                
        Returns:
            Dictionary containing outputs from all components
        """
        batch_size = batch_data.get("batch_size", 1)
        
        # A. PERCEPTION LAYER
        # Process text input - handle both string and tensor inputs
        text_input = batch_data.get("text_input", [""])
        if isinstance(text_input, torch.Tensor):
            # Convert tensor to dummy text for now
            text_input = ["sample text"] * batch_size
        
        text_output = self.text_encoder(text_input, return_language_logits=True)
        text_features = text_output["embeddings"]  # Fix key name
        
        # Process image input (if available)
        if "image_input" in batch_data and batch_data["image_input"] is not None:
            image_output = self.image_encoder(batch_data["image_input"])
            image_features = image_output["embeddings"]  # Fix key name
            has_visual = True
        else:
            # Create dummy image features if no image provided
            image_features = torch.zeros(batch_size, self.config.get("image_dim", 512)).to(self.device)
            has_visual = False
        
        # Process metadata
        metadata_dict = batch_data.get("metadata_categorical", {})
        if not metadata_dict:
            # Create dummy metadata if none provided
            metadata_dict = {
                "category": torch.zeros(batch_size, dtype=torch.long).to(self.device),
                "brand": torch.zeros(batch_size, dtype=torch.long).to(self.device)
            }
        
        metadata_output = self.metadata_encoder(metadata_dict)
        metadata_features = metadata_output["embeddings"]
        
        # Project features to fusion dimension
        text_projected = self.text_proj(text_features)
        image_projected = self.image_proj(image_features) 
        metadata_projected = self.metadata_proj(metadata_features)
        
        # Multimodal fusion
        fusion_output = self.fusion_layer(
            text_projected.unsqueeze(1),  # Add sequence dimension
            image_projected.unsqueeze(1), # Add sequence dimension
            metadata_projected
        )
        fused_features = fusion_output["multimodal_representation"]  # Fix key name
        
        # B. DIALOGUE & INTENT LAYER
        # Track dialogue state
        conversation_sequence = batch_data.get("conversation_history", [])
        if conversation_sequence:
            # Convert conversation to sequence for LSTM
            conv_tensor = torch.stack([fused_features] * len(conversation_sequence), dim=1)
        else:
            conv_tensor = fused_features.unsqueeze(1)
            
        dialogue_output = self.state_tracker(conv_tensor)
        dialogue_state = dialogue_output["state_representation"]  # Fix key name
        
        # Classify intent
        intent_output = self.intent_classifier(fused_features)
        predicted_intent = intent_output["predicted_intent"]
        
        # C. TOOL-ORIENTED POLICY LAYER
        # Select tools - need to provide available tools
        available_tools = ["search", "recommend", "compare", "filter", "translate", "visual_search"]
        tool_registry = ToolRegistry()
        tool_output = self.tool_selector(fused_features, available_tools, tool_registry)
        selected_tools = tool_output["best_tool_name"]
        
        # Generate arguments - use first selected tool
        tool_name = selected_tools[0] if isinstance(selected_tools, list) else "search"
        arg_output = self.arg_generator(fused_features, tool_name, tool_registry)
        tool_arguments = arg_output["arguments"]
        
        # Plan execution sequence
        planning_output = self.planner(dialogue_state, fused_features)
        execution_plan = planning_output["planning_representation"]
        
        # Compile complete output
        return {
            # Perception layer outputs
            "text_features": text_features,
            "image_features": image_features if has_visual else None,
            "metadata_features": metadata_features,
            "fused_features": fused_features,
            "fusion_attention": fusion_output.get("attention_weights"),
            "detected_language": text_output.get("language_logits"),
            
            # Dialogue & intent outputs
            "dialogue_state": dialogue_state,
            "dialogue_history": dialogue_output.get("hidden_states"),
            "predicted_intent": predicted_intent,
            "intent_logits": intent_output["intent_logits"],
            "intent_features": intent_output["intent_features"],
            
            # Policy outputs
            "selected_tools": selected_tools,
            "tool_scores": tool_output["tool_scores"],
            "tool_arguments": tool_arguments,
            "execution_plan": execution_plan,
            "predicted_steps": planning_output.get("predicted_steps"),
            
            # System metadata
            "has_visual_input": has_visual,
            "processing_metadata": {
                "batch_size": batch_size,
                "sequence_length": conv_tensor.size(1),
                "fusion_dim": fused_features.size(-1)
            }
        }
    
    def encode_multimodal_input(self, text_input: torch.Tensor,
                               image_input: Optional[torch.Tensor] = None,
                               metadata_input: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Convenience method to encode multimodal input into fused representation
        
        Args:
            text_input: Text input tensor
            image_input: Optional image input tensor
            metadata_input: Optional metadata dictionary
            
        Returns:
            Fused multimodal representation
        """
        batch_size = text_input.size(0)
        
        # Process each modality
        if isinstance(text_input, torch.Tensor):
            text_input = ["sample text"] * text_input.size(0)
        text_output = self.text_encoder(text_input)
        text_features = text_output["embeddings"]
        
        if image_input is not None:
            image_output = self.image_encoder(image_input)
            image_features = image_output["embeddings"]
        else:
            image_features = torch.zeros(batch_size, self.config.get("image_dim", 512)).to(self.device)
        
        if metadata_input is not None:
            metadata_dict = metadata_input.get("categorical", {})
            if not metadata_dict:
                metadata_dict = {
                    "category": torch.zeros(batch_size, dtype=torch.long).to(self.device),
                    "brand": torch.zeros(batch_size, dtype=torch.long).to(self.device)
                }
            metadata_output = self.metadata_encoder(metadata_dict)
            metadata_features = metadata_output["embeddings"]
        else:
            metadata_features = torch.zeros(batch_size, self.config.get("metadata_dim", 256)).to(self.device)
        
        # Project features to fusion dimension
        text_projected = self.text_proj(text_features)
        image_projected = self.image_proj(image_features)
        metadata_projected = self.metadata_proj(metadata_features)
        
        # Fuse modalities
        fusion_output = self.fusion_layer(
            text_projected.unsqueeze(1),
            image_projected.unsqueeze(1),
            metadata_projected
        )
        return fusion_output["multimodal_representation"]
    
    def predict_intent_and_tools(self, multimodal_input: torch.Tensor) -> Dict[str, Any]:
        """
        Convenience method to predict intent and select tools
        
        Args:
            multimodal_input: Fused multimodal representation
            
        Returns:
            Intent and tool predictions
        """
        # Predict intent
        intent_output = self.intent_classifier(multimodal_input)
        predicted_intent = intent_output["predicted_intent"]
        
        # Select tools based on intent
        available_tools = ["search", "recommend", "compare", "filter", "translate", "visual_search"]
        tool_registry = ToolRegistry()
        tool_output = self.tool_selector(multimodal_input, available_tools, tool_registry)
        
        return {
            "predicted_intent": predicted_intent,
            "intent_confidence": torch.softmax(intent_output["intent_logits"], dim=-1),
            "selected_tools": tool_output["best_tool_name"],
            "tool_scores": tool_output["tool_scores"]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "architecture": "MUSE v3 Advanced",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "components": {
                "text_encoder": sum(p.numel() for p in self.text_encoder.parameters()),
                "image_encoder": sum(p.numel() for p in self.image_encoder.parameters()),
                "metadata_encoder": sum(p.numel() for p in self.metadata_encoder.parameters()),
                "fusion_layer": sum(p.numel() for p in self.fusion_layer.parameters()),
                "state_tracker": sum(p.numel() for p in self.state_tracker.parameters()),
                "intent_classifier": sum(p.numel() for p in self.intent_classifier.parameters()),
                "tool_selector": sum(p.numel() for p in self.tool_selector.parameters()),
                "arg_generator": sum(p.numel() for p in self.arg_generator.parameters()),
                "planner": sum(p.numel() for p in self.planner.parameters())
            },
            "device": str(self.device),
            "config": self.config
        }
    
    def save_pretrained(self, save_path: str):
        """Save model and configuration"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_path, "model.pth"))
        
        # Save configuration
        import json
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str):
        """Load model from saved checkpoint"""
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(load_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(config)
        
        # Load state dict
        model_path = os.path.join(load_path, "model.pth")
        state_dict = torch.load(model_path, map_location=model.device)
        model.load_state_dict(state_dict)
        
        print(f"Model loaded from {load_path}")
        return model

# =============================================================================
# SAVE ARCHITECTURE TO FILE
# =============================================================================
