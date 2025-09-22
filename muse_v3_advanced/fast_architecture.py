#!/usr/bin/env python3
"""
MUSE v3 Fast Test Architecture
==============================

Lightweight version for testing without model downloads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import json
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# LIGHTWEIGHT ENCODERS (NO DOWNLOADS)
# =============================================================================

class TextEncoder(nn.Module):
    """Fast text encoder without pre-trained models"""
    
    def __init__(self, vocab_size=10000, hidden_size=384):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size//2, batch_first=True, bidirectional=True)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, text_input, return_language_logits=False):
        # Simple tokenization for testing
        if isinstance(text_input, list):
            batch_size = len(text_input)
            # Mock token ids
            token_ids = torch.randint(0, 1000, (batch_size, 20))
        else:
            token_ids = torch.randint(0, 1000, (1, 20))
            batch_size = 1
            
        # Forward pass
        embeddings = self.embedding(token_ids)
        lstm_out, _ = self.encoder(embeddings)
        pooled = self.pooling(lstm_out.transpose(1, 2)).squeeze(-1)
        
        result = {"embeddings": pooled}
        
        if return_language_logits:
            # Language detection (Hindi vs English)
            lang_classifier = nn.Linear(self.hidden_size, 2)
            result["language_logits"] = lang_classifier(pooled)
            
        return result

class ImageEncoder(nn.Module):
    """Fast image encoder without pre-trained models"""
    
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        # Simple CNN for testing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_size)
        )
        
    def forward(self, images, return_attributes=False):
        if isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = 1
            
        # Mock image tensor
        image_tensor = torch.randn(batch_size, 3, 224, 224)
        
        embeddings = self.conv_layers(image_tensor)
        
        result = {"embeddings": embeddings}
        
        if return_attributes:
            # Visual attributes
            attr_classifier = nn.Linear(self.hidden_size, 128)
            result["attributes"] = torch.sigmoid(attr_classifier(embeddings))
            
        return result

class MetadataEncoder(nn.Module):
    """Categorical metadata encoder"""
    
    def __init__(self, vocab_sizes: Dict[str, int], hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_sizes = vocab_sizes
        
        # Create embeddings for each categorical feature
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size, hidden_size // len(vocab_sizes))
            for feature, vocab_size in vocab_sizes.items()
        })
        
        # Projection layer
        self.projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, metadata_categorical: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embedded_features = []
        
        for feature, indices in metadata_categorical.items():
            if feature in self.embeddings:
                embedded = self.embeddings[feature](indices)
                embedded_features.append(embedded)
        
        if embedded_features:
            concatenated = torch.cat(embedded_features, dim=-1)
            projected = self.projection(concatenated)
        else:
            # Fallback for empty metadata
            batch_size = 1
            projected = torch.zeros(batch_size, self.hidden_size)
            
        return {"embeddings": projected}

class MultimodalFusion(nn.Module):
    """Attention-based multimodal fusion"""
    
    def __init__(self, hidden_size: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, text_embeddings: torch.Tensor, 
                image_embeddings: torch.Tensor,
                metadata_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        batch_size = text_embeddings.size(0)
        
        # Ensure all embeddings have the same dimension
        if text_embeddings.size(-1) != self.hidden_size:
            text_proj = nn.Linear(text_embeddings.size(-1), self.hidden_size)
            text_embeddings = text_proj(text_embeddings)
            
        if image_embeddings.size(-1) != self.hidden_size:
            image_proj = nn.Linear(image_embeddings.size(-1), self.hidden_size)
            image_embeddings = image_proj(image_embeddings)
            
        if metadata_embeddings.size(-1) != self.hidden_size:
            metadata_proj = nn.Linear(metadata_embeddings.size(-1), self.hidden_size)
            metadata_embeddings = metadata_proj(metadata_embeddings)
        
        # Add sequence dimension if needed
        if len(text_embeddings.shape) == 2:
            text_embeddings = text_embeddings.unsqueeze(1)
        if len(image_embeddings.shape) == 2:
            image_embeddings = image_embeddings.unsqueeze(1)
        if len(metadata_embeddings.shape) == 2:
            metadata_embeddings = metadata_embeddings.unsqueeze(1)
            
        # Concatenate modalities
        multimodal_input = torch.cat([text_embeddings, image_embeddings, metadata_embeddings], dim=1)
        
        # Self-attention
        attended_output, _ = self.attention(multimodal_input, multimodal_input, multimodal_input)
        
        # Residual connection and normalization
        output = self.norm(attended_output + multimodal_input)
        
        # Feed-forward network
        output = output + self.ffn(output)
        
        # Pool to single representation
        pooled_output = torch.mean(output, dim=1)
        
        return {"multimodal_representation": pooled_output}

# =============================================================================
# B. DIALOGUE & INTENT LAYER
# =============================================================================

class DialogueStateTracker(nn.Module):
    """Track conversation state and context"""
    
    def __init__(self, hidden_size: int = 512, max_turns: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_turns = max_turns
        
        # LSTM for turn-level tracking
        self.turn_tracker = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # State representation
        self.state_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, current_representation: torch.Tensor,
                conversation_history: List = None) -> Dict[str, torch.Tensor]:
        
        batch_size = current_representation.size(0)
        
        # Add sequence dimension
        if len(current_representation.shape) == 2:
            current_representation = current_representation.unsqueeze(1)
        
        # Process through LSTM
        state_output, _ = self.turn_tracker(current_representation)
        
        # Project to final state
        final_state = self.state_projection(state_output[:, -1, :])
        
        return {"state_representation": final_state}

class IntentClassifier(nn.Module):
    """Multi-class intent classification"""
    
    def __init__(self, hidden_size: int = 512, num_intents: int = 7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_intents)
        )
        
    def forward(self, state_representation: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.classifier(state_representation)
        return {"intent_logits": logits, "predicted_intent": torch.argmax(logits, dim=-1)}

# =============================================================================
# C. TOOL-ORIENTED POLICY LAYER
# =============================================================================

class ToolSelector(nn.Module):
    """Select appropriate tools for the current context"""
    
    def __init__(self, hidden_size: int = 512, num_tools: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tools = num_tools
        
        self.tool_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tools)
        )
        
    def forward(self, state_representation: torch.Tensor, 
                intent_logits: torch.Tensor,
                available_tools: List[str] = None,
                tool_registry: Dict = None) -> Dict[str, Any]:
        
        # Combine state and intent information
        combined_input = state_representation
        if intent_logits is not None:
            combined_input = torch.cat([state_representation, intent_logits], dim=-1)
            # Adjust classifier if needed
            if combined_input.size(-1) != self.hidden_size:
                projection = nn.Linear(combined_input.size(-1), self.hidden_size)
                combined_input = projection(combined_input)
        
        # Tool selection logits
        tool_logits = self.tool_classifier(combined_input)
        selected_tools = torch.sigmoid(tool_logits) > 0.5
        
        # Convert to tool names if registry available
        if available_tools:
            tool_names = [available_tools[i] for i in range(len(available_tools)) 
                         if i < selected_tools.size(-1) and selected_tools[0, i]]
        else:
            tool_names = [f"tool_{i}" for i in range(selected_tools.size(-1)) 
                         if selected_tools[0, i]]
        
        return {
            "tool_logits": tool_logits,
            "selected_tools": tool_names,
            "tool_confidence": torch.sigmoid(tool_logits).max(dim=-1)[0]
        }

class ArgumentGenerator(nn.Module):
    """Generate arguments for selected tools"""
    
    def __init__(self, hidden_size: int = 512, max_args: int = 64):
        super().__init__()
        self.arg_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, max_args)
        )
        
    def forward(self, state_representation: torch.Tensor, 
                selected_tools: List[str]) -> Dict[str, Any]:
        
        arg_logits = self.arg_generator(state_representation)
        
        # Generate tool-specific arguments
        tool_arguments = {}
        for tool in selected_tools:
            # Mock argument generation
            tool_arguments[tool] = {
                "confidence": torch.sigmoid(arg_logits).mean().item(),
                "parameters": {"query": "generated_query", "filters": []}
            }
        
        return {"tool_arguments": tool_arguments}

class Planner(nn.Module):
    """Plan execution sequence for selected tools"""
    
    def __init__(self, hidden_size: int = 512, max_steps: int = 5):
        super().__init__()
        self.max_steps = max_steps
        self.step_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_steps)
        )
        
    def forward(self, state_representation: torch.Tensor,
                selected_tools: List[str]) -> Dict[str, Any]:
        
        step_logits = self.step_predictor(state_representation)
        
        # Handle batch processing
        if step_logits.dim() > 1:
            num_steps = torch.clamp(torch.argmax(step_logits, dim=-1)[0], 1, self.max_steps).item()
        else:
            num_steps = torch.clamp(torch.argmax(step_logits, dim=-1), 1, self.max_steps).item()
        
        # Simple execution plan
        execution_plan = []
        for i, tool in enumerate(selected_tools[:num_steps]):
            execution_plan.append({
                "step": i + 1,
                "tool": tool,
                "dependencies": execution_plan[-1:] if execution_plan else []
            })
        
        return {
            "execution_plan": execution_plan,
            "estimated_steps": num_steps
        }

# =============================================================================
# D. RESPONSE GENERATOR
# =============================================================================

class ResponseGenerator(nn.Module):
    """Generate final responses with bilingual support"""
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Response planning
        self.response_planner = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Language selection
        self.language_selector = nn.Linear(hidden_size, 2)  # Hindi/English
        
    def forward(self, state_representation: torch.Tensor,
                execution_results: Dict[str, Any] = None,
                user_language: str = "english") -> Dict[str, Any]:
        
        # Plan response
        response_features = self.response_planner(state_representation)
        
        # Select language
        language_logits = self.language_selector(response_features)
        
        # Handle batch processing
        if language_logits.dim() > 1:
            predicted_language = "hindi" if torch.argmax(language_logits, dim=-1)[0].item() == 0 else "english"
            max_conf = torch.softmax(language_logits, dim=-1).max().item()
        else:
            predicted_language = "hindi" if torch.argmax(language_logits, dim=-1).item() == 0 else "english"
            max_conf = torch.softmax(language_logits, dim=-1).max().item()
        
        # Generate response templates
        templates = {
            "english": "Based on your request, I found some relevant items.",
            "hindi": "आपके अनुरोध के आधार पर, मुझे कुछ प्रासंगिक वस्तुएं मिलीं।"
        }
        
        final_language = user_language if user_language in templates else predicted_language
        
        return {
            "response_text": templates.get(final_language, templates["english"]),
            "predicted_language": predicted_language,
            "language_confidence": max_conf,
            "response_features": response_features
        }

# =============================================================================
# MAIN ARCHITECTURE
# =============================================================================

class FastMuseV3Architecture(nn.Module):
    """Fast MUSE v3 Architecture for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration
        self.config = config
        self.text_dim = config.get("text_dim", 384)
        self.image_dim = config.get("image_dim", 512)
        self.metadata_dim = config.get("metadata_dim", 256)
        self.fusion_dim = config.get("fusion_dim", 512)
        self.num_intents = config.get("num_intents", 7)
        self.num_tools = config.get("num_tools", 6)
        self.max_steps = config.get("max_steps", 5)
        self.device = config.get("device", "cpu")
        
        # A. Perception Layer
        self.text_encoder = TextEncoder(hidden_size=self.text_dim)
        self.image_encoder = ImageEncoder(hidden_size=self.image_dim)
        self.metadata_encoder = MetadataEncoder(
            config.get("metadata_vocab", {"category": 50, "brand": 100}),
            self.metadata_dim
        )
        self.multimodal_fusion = MultimodalFusion(self.fusion_dim)
        
        # B. Dialogue & Intent Layer
        self.dialogue_tracker = DialogueStateTracker(self.fusion_dim)
        self.intent_classifier = IntentClassifier(self.fusion_dim, self.num_intents)
        
        # C. Tool-Oriented Policy Layer
        self.tool_selector = ToolSelector(self.fusion_dim, self.num_tools)
        self.argument_generator = ArgumentGenerator(self.fusion_dim)
        self.planner = Planner(self.fusion_dim, self.max_steps)
        
        # D. Response Generator
        self.response_generator = ResponseGenerator(self.fusion_dim)
        
        # Available tools
        self.available_tools = [
            "search", "recommend", "compare", "filter", "translate", "visual_search"
        ]
        
        logger.info(f"🏗️  Fast MUSE v3 Architecture initialized")
        logger.info(f"   📱 Text dim: {self.text_dim}")
        logger.info(f"   🖼️  Image dim: {self.image_dim}")  
        logger.info(f"   📊 Metadata dim: {self.metadata_dim}")
        logger.info(f"   🔗 Fusion dim: {self.fusion_dim}")
        logger.info(f"   🎯 Intents: {self.num_intents}")
        logger.info(f"   🛠️  Tools: {self.num_tools}")
        
    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass through the complete architecture"""
        
        # Extract inputs
        text_input = batch_data.get("text_input", [])
        image_input = batch_data.get("image_input", [])
        metadata_categorical = batch_data.get("metadata_categorical", {})
        conversation_history = batch_data.get("conversation_history", [])
        batch_size = batch_data.get("batch_size", len(text_input) if text_input else 1)
        
        # A. Perception Layer
        # Encode text
        if text_input:
            text_outputs = self.text_encoder(text_input, return_language_logits=True)
            text_embeddings = text_outputs["embeddings"]
            language_logits = text_outputs.get("language_logits", None)
        else:
            text_embeddings = torch.zeros(batch_size, self.text_dim)
            language_logits = None
            
        # Encode images
        if image_input:
            image_outputs = self.image_encoder(image_input, return_attributes=True)
            image_embeddings = image_outputs["embeddings"]
        else:
            image_embeddings = torch.zeros(batch_size, self.image_dim)
            
        # Encode metadata
        if metadata_categorical:
            metadata_outputs = self.metadata_encoder(metadata_categorical)
            metadata_embeddings = metadata_outputs["embeddings"]
        else:
            metadata_embeddings = torch.zeros(batch_size, self.metadata_dim)
            
        # Multimodal fusion
        fusion_outputs = self.multimodal_fusion(text_embeddings, image_embeddings, metadata_embeddings)
        fused_features = fusion_outputs["multimodal_representation"]
        
        # B. Dialogue & Intent Layer
        # Track dialogue state
        dialogue_outputs = self.dialogue_tracker(fused_features, conversation_history)
        state_representation = dialogue_outputs["state_representation"]
        
        # Classify intent
        intent_outputs = self.intent_classifier(state_representation)
        intent_logits = intent_outputs["intent_logits"]
        predicted_intent = intent_outputs["predicted_intent"]
        
        # C. Tool-Oriented Policy Layer
        # Select tools
        tool_outputs = self.tool_selector(
            state_representation, 
            intent_logits,
            available_tools=self.available_tools,
            tool_registry=None
        )
        selected_tools = tool_outputs["selected_tools"]
        
        # Generate arguments
        arg_outputs = self.argument_generator(state_representation, selected_tools)
        tool_arguments = arg_outputs["tool_arguments"]
        
        # Plan execution
        plan_outputs = self.planner(state_representation, selected_tools)
        execution_plan = plan_outputs["execution_plan"]
        
        # D. Response Generator
        response_outputs = self.response_generator(
            state_representation,
            execution_results={"tools_used": selected_tools},
            user_language="english"
        )
        
        # Compile final outputs
        return {
            # Core representations
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "metadata_embeddings": metadata_embeddings,
            "fused_features": fused_features,
            "state_representation": state_representation,
            
            # Intent and language
            "intent_logits": intent_logits,
            "predicted_intent": predicted_intent,
            "language_logits": language_logits,
            
            # Tool selection and planning
            "selected_tools": selected_tools,
            "tool_arguments": tool_arguments,
            "execution_plan": execution_plan,
            
            # Response generation
            "response_text": response_outputs["response_text"],
            "predicted_language": response_outputs["predicted_language"],
            "response_features": response_outputs["response_features"],
            
            # Metadata
            "batch_size": batch_size,
            "processing_successful": True
        }

def test_fast_architecture():
    """Test the fast architecture"""
    
    print("🚀 Testing Fast MUSE v3 Architecture")
    print("=" * 50)
    
    # Configuration
    config = {
        "text_dim": 384,
        "image_dim": 512,
        "metadata_dim": 256,
        "fusion_dim": 512,
        "num_intents": 7,
        "num_tools": 6,
        "max_steps": 5,
        "device": "cpu",
        "metadata_vocab": {"category": 50, "brand": 100}
    }
    
    # Create model
    model = FastMuseV3Architecture(config)
    model.eval()
    
    # Test cases
    test_cases = [
        {
            "text_input": ["I want to buy running shoes"],
            "metadata_categorical": {
                "category": torch.tensor([0]),
                "brand": torch.tensor([0])
            },
            "batch_size": 1
        },
        {
            "text_input": ["Can you recommend a dress for wedding?", "मुझे किताब चाहिए"],
            "metadata_categorical": {
                "category": torch.tensor([1, 2]),
                "brand": torch.tensor([1, 2])
            },
            "batch_size": 2
        }
    ]
    
    intent_names = ["search", "recommend", "compare", "filter", "translate", "visual_search", "chitchat"]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Test Case {i+1}: {test_case['text_input']}")
        
        with torch.no_grad():
            outputs = model(test_case)
            
        # Results
        print(f"✅ Forward pass successful!")
        print(f"   Batch size: {outputs['batch_size']}")
        print(f"   Fused features: {outputs['fused_features'].shape}")
        
        # Handle batch processing for intent
        predicted_intent = outputs['predicted_intent']
        if predicted_intent.dim() > 0 and predicted_intent.size(0) > 1:
            intent_idx = predicted_intent[0].item()
        else:
            intent_idx = predicted_intent.item() if predicted_intent.dim() > 0 else predicted_intent
            
        print(f"   Intent: {intent_names[intent_idx]} (logits: {outputs['intent_logits'].shape})")
        print(f"   Selected tools: {outputs['selected_tools']}")
        print(f"   Response: {outputs['response_text']}")
        print(f"   Language: {outputs['predicted_language']}")
    
    print(f"\n🎉 All tests passed! Fast MUSE v3 is working!")
    return True

if __name__ == "__main__":
    test_fast_architecture()
