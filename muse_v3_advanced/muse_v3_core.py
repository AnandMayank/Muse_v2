#!/usr/bin/env python3
"""
MUSE v3 Advanced Multimodal System
==================================

Complete implementation with:
- Perception Layers (Text, Image, Metadata)
- Tool-Oriented Policy (OctoTools-inspired)
- Cross-lingual Support
- Multimodal Reasoning
- Advanced Training Pipeline
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import os
from dataclasses import dataclass
from transformers import (
    AutoModel, AutoTokenizer, 
    CLIPModel, CLIPProcessor,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
import faiss
from PIL import Image
import base64
from io import BytesIO

# =============================================================================
# A. PERCEPTION LAYER
# =============================================================================

class TextEncoder(nn.Module):
    """Advanced text encoder with T5/LLaMA-style architecture"""
    
    def __init__(self, model_name="google/flan-t5-base", hidden_size=768):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # Projection layer for consistent dimensions
        self.projection = nn.Linear(self.model.config.d_model, hidden_size)
        
    def encode_dialogue_context(self, dialogue_history: List[Dict]) -> torch.Tensor:
        """Encode dialogue context with persona and history"""
        
        # Format dialogue context
        context_text = self._format_dialogue_context(dialogue_history)
        
        # Tokenize
        inputs = self.tokenizer(
            context_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )
        
        # Get encoder outputs
        with torch.no_grad():
            outputs = self.model.encoder(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Global pooling and projection
        pooled = torch.mean(hidden_states, dim=1)
        projected = self.projection(pooled)
        
        return projected
    
    def _format_dialogue_context(self, dialogue_history: List[Dict]) -> str:
        """Format dialogue history with metadata"""
        
        context_parts = []
        
        for turn in dialogue_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            metadata = turn.get("metadata", {})
            
            # Add metadata context
            if metadata:
                persona = metadata.get("persona", "")
                preferences = metadata.get("preferences", [])
                if persona:
                    context_parts.append(f"[PERSONA: {persona}]")
                if preferences:
                    context_parts.append(f"[PREFERENCES: {', '.join(preferences)}]")
            
            context_parts.append(f"{role.upper()}: {content}")
        
        return " ".join(context_parts)

class ImageEncoder(nn.Module):
    """CLIP/BLIP-based image encoder for item images"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", hidden_size=768):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # Projection layer
        clip_dim = self.model.config.vision_config.hidden_size
        self.projection = nn.Linear(clip_dim, hidden_size)
        
        # Image similarity computation
        self.similarity_threshold = 0.7
        
    def encode_images(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """Encode multiple product images"""
        
        processed_images = []
        for img in images:
            if isinstance(img, str):
                # Handle base64 encoded images
                if img.startswith("data:image"):
                    img_data = img.split(",")[1]
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(BytesIO(img_bytes))
                else:
                    # Handle file paths
                    img = Image.open(img)
            processed_images.append(img)
        
        # Process images
        inputs = self.processor(images=processed_images, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_embeddings = vision_outputs.pooler_output
        
        # Project to consistent dimension
        projected = self.projection(image_embeddings)
        
        return projected
    
    def find_similar_items(self, query_image: Union[str, Image.Image], 
                          item_database: torch.Tensor) -> List[Tuple[int, float]]:
        """Find similar items using image similarity"""
        
        query_embedding = self.encode_images([query_image])
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(1), 
            item_database.unsqueeze(0), 
            dim=-1
        )
        
        # Get top similar items
        top_k = min(10, len(similarities[0]))
        top_similarities, top_indices = torch.topk(similarities[0], top_k)
        
        results = []
        for idx, sim in zip(top_indices, top_similarities):
            if sim.item() > self.similarity_threshold:
                results.append((idx.item(), sim.item()))
        
        return results

class MetadataEncoder(nn.Module):
    """Structured encoder for item attributes and metadata"""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Categorical embeddings
        self.category_embedding = nn.Embedding(1000, hidden_size // 4)  # Max 1000 categories
        self.brand_embedding = nn.Embedding(500, hidden_size // 4)      # Max 500 brands
        
        # Numerical feature processing
        self.numerical_projection = nn.Linear(10, hidden_size // 4)  # Price, rating, etc.
        
        # Text feature processing (for descriptions)
        self.text_projection = nn.Linear(768, hidden_size // 4)
        
        # Final projection
        self.final_projection = nn.Linear(hidden_size, hidden_size)
        
    def encode_metadata(self, metadata_batch: List[Dict]) -> torch.Tensor:
        """Encode structured metadata for items"""
        
        batch_embeddings = []
        
        for metadata in metadata_batch:
            # Categorical features
            category_id = metadata.get("category_id", 0)
            brand_id = metadata.get("brand_id", 0)
            
            cat_emb = self.category_embedding(torch.tensor(category_id))
            brand_emb = self.brand_embedding(torch.tensor(brand_id))
            
            # Numerical features
            numerical_features = torch.tensor([
                metadata.get("price", 0.0),
                metadata.get("rating", 0.0),
                metadata.get("num_reviews", 0.0),
                metadata.get("discount", 0.0),
                metadata.get("availability", 1.0),
                metadata.get("popularity_score", 0.0),
                metadata.get("seasonal_relevance", 0.0),
                metadata.get("trend_score", 0.0),
                metadata.get("quality_score", 0.0),
                metadata.get("sustainability_score", 0.0)
            ], dtype=torch.float32)
            
            num_emb = self.numerical_projection(numerical_features)
            
            # Text features (description embeddings - would be pre-computed)
            text_features = torch.zeros(768)  # Placeholder for description embeddings
            text_emb = self.text_projection(text_features)
            
            # Concatenate all embeddings
            combined = torch.cat([cat_emb, brand_emb, num_emb, text_emb], dim=-1)
            final_emb = self.final_projection(combined)
            
            batch_embeddings.append(final_emb)
        
        return torch.stack(batch_embeddings)

class MultimodalFusion(nn.Module):
    """Cross-attention fusion for multimodal understanding"""
    
    def __init__(self, hidden_size=768, num_heads=8, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Cross-attention layers
        self.text_to_image_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.image_to_text_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.metadata_fusion = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size * 3, hidden_size)
        
    def forward(self, text_embeddings: torch.Tensor, 
                image_embeddings: torch.Tensor,
                metadata_embeddings: torch.Tensor) -> torch.Tensor:
        """Fuse multimodal representations"""
        
        # Cross-attention between modalities
        text_attended, _ = self.text_to_image_attention(
            text_embeddings, image_embeddings, image_embeddings
        )
        
        image_attended, _ = self.image_to_text_attention(
            image_embeddings, text_embeddings, text_embeddings
        )
        
        metadata_attended, _ = self.metadata_fusion(
            metadata_embeddings, text_embeddings, text_embeddings
        )
        
        # Combine representations
        combined = torch.cat([text_attended, image_attended, metadata_attended], dim=-1)
        fused = self.output_projection(combined)
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            fused = layer(fused, fused)
        
        return fused

# =============================================================================
# B. DIALOGUE & INTENT LAYER
# =============================================================================

@dataclass
class DialogueState:
    """Dialogue state representation"""
    user_goals: List[str]
    constraints: Dict[str, Any]
    preferences: Dict[str, Any]
    persona: Dict[str, Any]
    conversation_history: List[Dict]
    current_intent: str
    language: str = "en"
    multimodal_context: Dict[str, Any] = None

class DialogueStateTracker(nn.Module):
    """Advanced dialogue state tracker"""
    
    def __init__(self, hidden_size=768, num_slots=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        
        # State representation
        self.state_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.slot_classifier = nn.Linear(hidden_size, num_slots * 2)  # Binary for each slot
        
        # Persona tracking
        self.persona_tracker = nn.Linear(hidden_size, 256)
        
    def update_state(self, current_state: DialogueState, 
                    new_turn: Dict, 
                    multimodal_features: torch.Tensor) -> DialogueState:
        """Update dialogue state with new turn"""
        
        # Encode current state
        state_vector = self._encode_state(current_state, multimodal_features)
        
        # Update slots
        slot_logits = self.slot_classifier(state_vector)
        slot_probabilities = torch.sigmoid(slot_logits)
        
        # Update persona
        persona_vector = self.persona_tracker(state_vector)
        
        # Extract updates
        updated_constraints = self._extract_constraints(new_turn, slot_probabilities)
        updated_preferences = self._extract_preferences(new_turn, persona_vector)
        
        # Create updated state
        updated_state = DialogueState(
            user_goals=current_state.user_goals + self._extract_goals(new_turn),
            constraints={**current_state.constraints, **updated_constraints},
            preferences={**current_state.preferences, **updated_preferences},
            persona=current_state.persona,
            conversation_history=current_state.conversation_history + [new_turn],
            current_intent=self._predict_intent(new_turn, multimodal_features),
            language=self._detect_language(new_turn),
            multimodal_context=new_turn.get("multimodal_context")
        )
        
        return updated_state
    
    def _encode_state(self, state: DialogueState, features: torch.Tensor) -> torch.Tensor:
        """Encode current dialogue state"""
        # Simplified encoding - in practice would be more sophisticated
        return torch.mean(features, dim=0)
    
    def _extract_constraints(self, turn: Dict, slot_probs: torch.Tensor) -> Dict[str, Any]:
        """Extract constraints from user turn"""
        constraints = {}
        content = turn.get("content", "").lower()
        
        # Price constraints
        if "under" in content or "below" in content:
            # Extract price constraint
            constraints["max_price"] = self._extract_price(content)
        
        # Category constraints
        categories = ["clothing", "electronics", "books", "home", "sports"]
        for cat in categories:
            if cat in content:
                constraints["category"] = cat
        
        return constraints
    
    def _extract_preferences(self, turn: Dict, persona_vector: torch.Tensor) -> Dict[str, Any]:
        """Extract user preferences"""
        preferences = {}
        content = turn.get("content", "").lower()
        
        # Style preferences
        styles = ["casual", "formal", "vintage", "modern", "minimalist"]
        for style in styles:
            if style in content:
                preferences["style"] = style
        
        # Color preferences
        colors = ["red", "blue", "green", "black", "white", "brown"]
        for color in colors:
            if color in content:
                preferences["preferred_color"] = color
        
        return preferences
    
    def _extract_goals(self, turn: Dict) -> List[str]:
        """Extract user goals from turn"""
        goals = []
        content = turn.get("content", "").lower()
        
        goal_keywords = {
            "buy": "purchase",
            "find": "search", 
            "recommend": "recommendation",
            "compare": "comparison",
            "show": "browse"
        }
        
        for keyword, goal in goal_keywords.items():
            if keyword in content:
                goals.append(goal)
        
        return goals
    
    def _predict_intent(self, turn: Dict, features: torch.Tensor) -> str:
        """Predict user intent"""
        content = turn.get("content", "").lower()
        
        # Rule-based intent classification (would be ML-based in practice)
        if any(word in content for word in ["find", "search", "looking for"]):
            return "search"
        elif any(word in content for word in ["recommend", "suggest"]):
            return "recommendation"
        elif any(word in content for word in ["compare", "difference"]):
            return "comparison"
        elif any(word in content for word in ["show", "see", "browse"]):
            return "browse"
        elif any(word in content for word in ["hello", "hi", "how are you"]):
            return "chitchat"
        else:
            return "general"
    
    def _detect_language(self, turn: Dict) -> str:
        """Detect language of user input"""
        content = turn.get("content", "")
        
        # Simple language detection (would use proper library in practice)
        hindi_chars = any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in content)
        if hindi_chars:
            return "hi"
        else:
            return "en"
    
    def _extract_price(self, text: str) -> float:
        """Extract price from text"""
        import re
        price_match = re.search(r'(\d+(?:\.\d{2})?)', text)
        return float(price_match.group(1)) if price_match else 0.0

class IntentClassifier(nn.Module):
    """Advanced intent classifier for dialogue turns"""
    
    def __init__(self, hidden_size=768, num_intents=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_intents = num_intents
        
        # Intent categories
        self.intent_labels = [
            "search", "filter", "compare", "recommend", 
            "chitchat", "clarification", "confirmation",
            "browse", "purchase", "general"
        ]
        
        # Classification head
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_intents)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Linear(hidden_size, 1)
        
    def classify_intent(self, text_features: torch.Tensor, 
                       multimodal_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Classify user intent with confidence"""
        
        # Combine features if multimodal
        if multimodal_features is not None:
            combined_features = torch.cat([text_features, multimodal_features], dim=-1)
            # Add projection layer if dimensions don't match
            if combined_features.size(-1) != self.hidden_size:
                projection = nn.Linear(combined_features.size(-1), self.hidden_size)
                combined_features = projection(combined_features)
        else:
            combined_features = text_features
        
        # Predict intent
        intent_logits = self.intent_classifier(combined_features)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        
        # Predict confidence
        confidence = torch.sigmoid(self.confidence_head(combined_features))
        
        # Get top intent
        top_intent_idx = torch.argmax(intent_probs, dim=-1)
        top_intent = self.intent_labels[top_intent_idx.item()]
        top_prob = intent_probs[0, top_intent_idx].item()
        
        return {
            "intent": top_intent,
            "confidence": confidence.item(),
            "probability": top_prob,
            "all_probabilities": {
                label: prob.item() 
                for label, prob in zip(self.intent_labels, intent_probs[0])
            }
        }

# =============================================================================
# C. TOOL-ORIENTED POLICY LAYER (OctoTools-inspired)
# =============================================================================

class ToolSelector(nn.Module):
    """Tool selector head for choosing optimal tools"""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Available tools
        self.available_tools = [
            "search_items",
            "filter_items", 
            "recommend_similar",
            "compare_items",
            "get_item_details",
            "translate_text",
            "visual_search",
            "price_comparison",
            "availability_check",
            "user_profile_update"
        ]
        
        # Tool selection network
        self.tool_selector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # State + multimodal features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, len(self.available_tools))
        )
        
        # Tool relevance scoring
        self.relevance_scorer = nn.Linear(hidden_size, 1)
        
    def select_tools(self, dialogue_state: DialogueState, 
                    multimodal_features: torch.Tensor,
                    max_tools: int = 3) -> List[Dict[str, Any]]:
        """Select best tools for current dialogue state"""
        
        # Encode dialogue state
        state_vector = self._encode_dialogue_state(dialogue_state)
        
        # Combine with multimodal features
        combined_features = torch.cat([state_vector, multimodal_features.flatten()], dim=-1)
        
        # Select tools
        tool_logits = self.tool_selector(combined_features)
        tool_probabilities = torch.softmax(tool_logits, dim=-1)
        
        # Get relevance scores
        relevance = torch.sigmoid(self.relevance_scorer(state_vector))
        
        # Select top tools
        top_tools = []
        tool_probs_sorted, indices = torch.sort(tool_probabilities, descending=True)
        
        for i in range(min(max_tools, len(indices))):
            tool_idx = indices[i].item()
            tool_name = self.available_tools[tool_idx]
            probability = tool_probs_sorted[i].item()
            
            # Only select if above threshold
            if probability > 0.1:
                top_tools.append({
                    "tool": tool_name,
                    "probability": probability,
                    "relevance": relevance.item(),
                    "reason": self._get_selection_reason(tool_name, dialogue_state)
                })
        
        return top_tools
    
    def _encode_dialogue_state(self, state: DialogueState) -> torch.Tensor:
        """Encode dialogue state for tool selection"""
        # Simplified encoding - would use more sophisticated method
        
        # Intent encoding
        intent_vector = torch.zeros(self.hidden_size // 4)
        if state.current_intent == "search":
            intent_vector[0] = 1.0
        elif state.current_intent == "recommend":
            intent_vector[1] = 1.0
        elif state.current_intent == "compare":
            intent_vector[2] = 1.0
        
        # Constraints encoding
        constraints_vector = torch.zeros(self.hidden_size // 4)
        if state.constraints:
            constraints_vector[0] = len(state.constraints) / 10.0  # Normalized
        
        # Goals encoding
        goals_vector = torch.zeros(self.hidden_size // 4)
        if state.user_goals:
            goals_vector[0] = len(state.user_goals) / 5.0  # Normalized
        
        # Language encoding
        lang_vector = torch.zeros(self.hidden_size // 4)
        if state.language == "hi":
            lang_vector[0] = 1.0
        elif state.language == "en":
            lang_vector[1] = 1.0
        
        return torch.cat([intent_vector, constraints_vector, goals_vector, lang_vector])
    
    def _get_selection_reason(self, tool_name: str, state: DialogueState) -> str:
        """Get reason for tool selection"""
        
        reasons = {
            "search_items": f"User intent is '{state.current_intent}' which requires searching",
            "filter_items": f"User has {len(state.constraints)} constraints to apply",
            "recommend_similar": "User is seeking recommendations",
            "compare_items": "User wants to compare options",
            "translate_text": f"User is speaking in {state.language}",
            "visual_search": "Multimodal context available for visual search"
        }
        
        return reasons.get(tool_name, "Tool selected based on current context")

class ArgumentGenerator(nn.Module):
    """Generate arguments for selected tools"""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Argument extraction networks
        self.category_predictor = nn.Linear(hidden_size, 100)  # 100 categories
        self.price_range_predictor = nn.Linear(hidden_size, 2)  # min, max price
        self.text_query_generator = nn.Linear(hidden_size, hidden_size)
        
    def generate_arguments(self, tool_name: str, 
                          dialogue_state: DialogueState,
                          multimodal_features: torch.Tensor) -> Dict[str, Any]:
        """Generate arguments for a specific tool"""
        
        state_vector = self._encode_state_for_args(dialogue_state)
        
        if tool_name == "search_items":
            return self._generate_search_args(state_vector, dialogue_state)
        elif tool_name == "filter_items":
            return self._generate_filter_args(state_vector, dialogue_state)
        elif tool_name == "recommend_similar":
            return self._generate_recommendation_args(state_vector, dialogue_state)
        elif tool_name == "translate_text":
            return self._generate_translation_args(dialogue_state)
        elif tool_name == "visual_search":
            return self._generate_visual_search_args(multimodal_features, dialogue_state)
        else:
            return {"query": "general search"}
    
    def _generate_search_args(self, state_vector: torch.Tensor, 
                            state: DialogueState) -> Dict[str, Any]:
        """Generate search arguments"""
        
        # Extract search query from conversation
        last_turn = state.conversation_history[-1] if state.conversation_history else {}
        query = last_turn.get("content", "")
        
        # Predict category
        category_logits = self.category_predictor(state_vector)
        category_id = torch.argmax(category_logits).item()
        
        # Price range from constraints
        price_min = state.constraints.get("min_price", 0)
        price_max = state.constraints.get("max_price", 10000)
        
        return {
            "query": query,
            "category_id": category_id,
            "price_min": price_min,
            "price_max": price_max,
            "limit": 10
        }
    
    def _generate_filter_args(self, state_vector: torch.Tensor,
                            state: DialogueState) -> Dict[str, Any]:
        """Generate filter arguments"""
        
        filters = {}
        
        # Apply constraints as filters
        for key, value in state.constraints.items():
            filters[key] = value
        
        # Apply preferences as filters
        for key, value in state.preferences.items():
            filters[key] = value
        
        return {"filters": filters}
    
    def _generate_recommendation_args(self, state_vector: torch.Tensor,
                                    state: DialogueState) -> Dict[str, Any]:
        """Generate recommendation arguments"""
        
        return {
            "user_preferences": state.preferences,
            "conversation_context": state.conversation_history[-3:],  # Last 3 turns
            "persona": state.persona,
            "num_recommendations": 5
        }
    
    def _generate_translation_args(self, state: DialogueState) -> Dict[str, Any]:
        """Generate translation arguments"""
        
        last_turn = state.conversation_history[-1] if state.conversation_history else {}
        text_to_translate = last_turn.get("content", "")
        
        source_lang = state.language
        target_lang = "en" if source_lang == "hi" else "hi"
        
        return {
            "text": text_to_translate,
            "source_language": source_lang,
            "target_language": target_lang
        }
    
    def _generate_visual_search_args(self, multimodal_features: torch.Tensor,
                                   state: DialogueState) -> Dict[str, Any]:
        """Generate visual search arguments"""
        
        return {
            "image_features": multimodal_features.tolist(),
            "similarity_threshold": 0.7,
            "max_results": 10,
            "include_metadata": True
        }
    
    def _encode_state_for_args(self, state: DialogueState) -> torch.Tensor:
        """Encode state for argument generation"""
        return torch.randn(self.hidden_size)  # Placeholder

class ToolPlanner(nn.Module):
    """Multi-step tool planning"""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Planning network
        self.planner = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.action_predictor = nn.Linear(hidden_size, 20)  # Max 20 different actions
        
        # Plan types
        self.plan_types = [
            "single_tool",
            "sequential_tools", 
            "parallel_tools",
            "conditional_tools"
        ]
        
    def create_execution_plan(self, selected_tools: List[Dict], 
                            dialogue_state: DialogueState) -> Dict[str, Any]:
        """Create execution plan for selected tools"""
        
        if len(selected_tools) == 1:
            return self._create_single_tool_plan(selected_tools[0])
        elif len(selected_tools) > 1:
            return self._create_multi_tool_plan(selected_tools, dialogue_state)
        else:
            return {"plan_type": "no_action", "steps": []}
    
    def _create_single_tool_plan(self, tool_info: Dict) -> Dict[str, Any]:
        """Create plan for single tool execution"""
        
        return {
            "plan_type": "single_tool",
            "steps": [{
                "step_id": 1,
                "tool": tool_info["tool"],
                "action": "execute",
                "dependencies": [],
                "expected_output": "tool_results"
            }],
            "estimated_time": 500,  # milliseconds
            "fallback_plan": self._create_fallback_plan(tool_info["tool"])
        }
    
    def _create_multi_tool_plan(self, tools: List[Dict], 
                              state: DialogueState) -> Dict[str, Any]:
        """Create plan for multiple tools"""
        
        # Determine execution strategy
        if self._should_execute_parallel(tools, state):
            return self._create_parallel_plan(tools)
        else:
            return self._create_sequential_plan(tools)
    
    def _should_execute_parallel(self, tools: List[Dict], 
                               state: DialogueState) -> bool:
        """Determine if tools can be executed in parallel"""
        
        # Tools that can run in parallel
        parallel_safe = [
            "search_items", "filter_items", "get_item_details",
            "price_comparison", "availability_check"
        ]
        
        tool_names = [t["tool"] for t in tools]
        
        # All tools must be parallel-safe
        return all(tool in parallel_safe for tool in tool_names)
    
    def _create_parallel_plan(self, tools: List[Dict]) -> Dict[str, Any]:
        """Create parallel execution plan"""
        
        steps = []
        for i, tool_info in enumerate(tools):
            steps.append({
                "step_id": i + 1,
                "tool": tool_info["tool"],
                "action": "execute_parallel",
                "dependencies": [],
                "expected_output": f"tool_results_{i}"
            })
        
        return {
            "plan_type": "parallel_tools",
            "steps": steps,
            "estimated_time": max(300, 100 * len(tools)),  # Parallel execution time
            "merge_results": True
        }
    
    def _create_sequential_plan(self, tools: List[Dict]) -> Dict[str, Any]:
        """Create sequential execution plan"""
        
        steps = []
        for i, tool_info in enumerate(tools):
            dependencies = [f"tool_results_{i-1}"] if i > 0 else []
            
            steps.append({
                "step_id": i + 1,
                "tool": tool_info["tool"],
                "action": "execute_sequential",
                "dependencies": dependencies,
                "expected_output": f"tool_results_{i}"
            })
        
        return {
            "plan_type": "sequential_tools",
            "steps": steps,
            "estimated_time": 200 * len(tools),  # Sequential execution time
            "early_termination": True  # Can stop if sufficient results
        }
    
    def _create_fallback_plan(self, primary_tool: str) -> Dict[str, Any]:
        """Create fallback plan if primary tool fails"""
        
        fallback_tools = {
            "search_items": "filter_items",
            "visual_search": "search_items", 
            "recommend_similar": "search_items",
            "translate_text": "general_response"
        }
        
        fallback_tool = fallback_tools.get(primary_tool, "general_response")
        
        return {
            "fallback_tool": fallback_tool,
            "fallback_message": "I'll try a different approach to help you.",
            "retry_attempts": 2
        }

class CrossLingualTools:
    """Cross-lingual and bilingual tool implementations"""
    
    def __init__(self):
        # Initialize translation pipeline
        try:
            self.translator = pipeline(
                "translation", 
                model="Helsinki-NLP/opus-mt-en-hi",
                return_all_scores=True
            )
            self.reverse_translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-hi-en"
            )
        except:
            self.translator = None
            self.reverse_translator = None
            print("Translation models not available - using mock translations")
        
        # Language detection
        self.language_patterns = {
            "hi": [
                "क्या", "कैसे", "कहाँ", "कब", "क्यों", "है", "हैं", 
                "मुझे", "आप", "यह", "वह", "के", "में", "से"
            ],
            "en": [
                "what", "how", "where", "when", "why", "is", "are",
                "me", "you", "this", "that", "the", "in", "of"
            ]
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        
        text_lower = text.lower()
        
        # Count language-specific patterns
        hi_score = sum(1 for pattern in self.language_patterns["hi"] if pattern in text)
        en_score = sum(1 for pattern in self.language_patterns["en"] if pattern in text_lower)
        
        # Also check for Unicode ranges
        hindi_chars = sum(1 for char in text if 0x0900 <= ord(char) <= 0x097F)
        
        if hindi_chars > 0 or hi_score > en_score:
            return "hi"
        else:
            return "en"
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Translate text between languages"""
        
        if source_lang == target_lang:
            return {
                "translated_text": text,
                "confidence": 1.0,
                "source_language": source_lang,
                "target_language": target_lang
            }
        
        if self.translator is None:
            # Mock translation for demo
            return self._mock_translate(text, source_lang, target_lang)
        
        try:
            if source_lang == "en" and target_lang == "hi":
                result = self.translator(text)
            elif source_lang == "hi" and target_lang == "en":
                result = self.reverse_translator(text)
            else:
                # Unsupported language pair
                return self._mock_translate(text, source_lang, target_lang)
            
            return {
                "translated_text": result[0]["translation_text"],
                "confidence": result[0].get("score", 0.8),
                "source_language": source_lang,
                "target_language": target_lang
            }
        
        except Exception as e:
            print(f"Translation error: {e}")
            return self._mock_translate(text, source_lang, target_lang)
    
    def _mock_translate(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Mock translation for demonstration"""
        
        mock_translations = {
            ("en", "hi"): {
                "hello": "नमस्ते",
                "how are you": "आप कैसे हैं",
                "thank you": "धन्यवाद",
                "good": "अच्छा",
                "bad": "बुरा",
                "yes": "हाँ",
                "no": "नहीं",
                "I need": "मुझे चाहिए",
                "show me": "मुझे दिखाएं",
                "recommend": "सुझाएं"
            },
            ("hi", "en"): {
                "नमस्ते": "hello",
                "आप कैसे हैं": "how are you", 
                "धन्यवाद": "thank you",
                "अच्छा": "good",
                "बुरा": "bad",
                "हाँ": "yes",
                "नहीं": "no",
                "मुझे चाहिए": "I need",
                "मुझे दिखाएं": "show me",
                "सुझाएं": "recommend"
            }
        }
        
        translations = mock_translations.get((source_lang, target_lang), {})
        
        # Simple word-by-word translation
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            translated = translations.get(word, word)  # Keep original if no translation
            translated_words.append(translated)
        
        translated_text = " ".join(translated_words)
        
        return {
            "translated_text": translated_text,
            "confidence": 0.7,  # Mock confidence
            "source_language": source_lang,
            "target_language": target_lang,
            "note": "Mock translation for demonstration"
        }
    
    def create_bilingual_response(self, response_en: str, target_lang: str = "hi") -> Dict[str, Any]:
        """Create bilingual response"""
        
        if target_lang == "en":
            return {
                "primary_response": response_en,
                "secondary_response": None,
                "language": "en"
            }
        
        # Translate to target language
        translation_result = self.translate_text(response_en, "en", target_lang)
        
        return {
            "primary_response": translation_result["translated_text"],
            "secondary_response": response_en,  # Keep English as secondary
            "language": target_lang,
            "bilingual": True,
            "translation_confidence": translation_result["confidence"]
        }

# =============================================================================
# D. RESPONSE GENERATOR
# =============================================================================

class ResponseGenerator(nn.Module):
    """Advanced response generator with bilingual support"""
    
    def __init__(self, model_name="google/flan-t5-base", hidden_size=768):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # Cross-lingual tools
        self.cross_lingual = CrossLingualTools()
        
        # Response templates
        self.templates = BilingualTemplates()
        
    def generate_response(self, dialogue_state: DialogueState,
                         tool_results: Dict[str, Any],
                         multimodal_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Generate natural, grounded response"""
        
        # Detect response language
        response_language = dialogue_state.language
        
        # Create context for generation
        context = self._create_generation_context(dialogue_state, tool_results)
        
        # Generate base response
        if tool_results:
            response = self._generate_grounded_response(context, tool_results, response_language)
        else:
            response = self._generate_conversational_response(context, response_language)
        
        # Add multimodal elements if available
        if multimodal_features is not None and dialogue_state.multimodal_context:
            response = self._enhance_with_multimodal(response, dialogue_state.multimodal_context)
        
        # Create bilingual version if needed
        if response_language != "en":
            bilingual_response = self.cross_lingual.create_bilingual_response(
                response, response_language
            )
            response = bilingual_response["primary_response"]
        
        return {
            "response": response,
            "language": response_language,
            "confidence": 0.9,
            "response_type": self._classify_response_type(dialogue_state, tool_results),
            "multimodal_enhanced": multimodal_features is not None,
            "tool_grounded": bool(tool_results)
        }
    
    def _create_generation_context(self, state: DialogueState, 
                                 tool_results: Dict[str, Any]) -> str:
        """Create context for response generation"""
        
        context_parts = []
        
        # Add user intent
        context_parts.append(f"Intent: {state.current_intent}")
        
        # Add constraints and preferences
        if state.constraints:
            constraints_str = ", ".join([f"{k}: {v}" for k, v in state.constraints.items()])
            context_parts.append(f"Constraints: {constraints_str}")
        
        if state.preferences:
            prefs_str = ", ".join([f"{k}: {v}" for k, v in state.preferences.items()])
            context_parts.append(f"Preferences: {prefs_str}")
        
        # Add recent conversation
        if state.conversation_history:
            recent_turns = state.conversation_history[-2:]  # Last 2 turns
            for turn in recent_turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                context_parts.append(f"{role}: {content}")
        
        # Add tool results summary
        if tool_results:
            context_parts.append(f"Tool Results: {self._summarize_tool_results(tool_results)}")
        
        return "\n".join(context_parts)
    
    def _generate_grounded_response(self, context: str, tool_results: Dict[str, Any],
                                  language: str) -> str:
        """Generate response grounded in tool results"""
        
        # Use templates for consistent responses
        template = self.templates.get_template("recommendation", language)
        
        # Extract key information from tool results
        items = tool_results.get("items", [])
        
        if items:
            # Format items for response
            if len(items) == 1:
                item = items[0]
                response = template["single_item"].format(
                    item_name=item.get("name", "item"),
                    price=item.get("price", "N/A"),
                    description=item.get("description", "")[:100]
                )
            else:
                # Multiple items
                item_list = []
                for i, item in enumerate(items[:3]):  # Top 3
                    item_list.append(f"{i+1}. {item.get('name', 'item')} - ${item.get('price', 'N/A')}")
                
                response = template["multiple_items"].format(
                    item_count=len(items),
                    item_list="\n".join(item_list)
                )
        else:
            response = template["no_results"]
        
        return response
    
    def _generate_conversational_response(self, context: str, language: str) -> str:
        """Generate conversational response without tools"""
        
        template = self.templates.get_template("general", language)
        
        # Simple rule-based responses
        if "hello" in context.lower() or "hi" in context.lower():
            return template["greeting"]
        elif "thank" in context.lower():
            return template["acknowledgment"]
        else:
            return template["general_help"]
    
    def _enhance_with_multimodal(self, response: str, multimodal_context: Dict) -> str:
        """Enhance response with multimodal context"""
        
        if multimodal_context.get("has_images"):
            image_info = multimodal_context.get("image_description", "")
            if image_info:
                response += f"\n\nBased on the image you shared ({image_info}), {response.lower()}"
        
        return response
    
    def _summarize_tool_results(self, tool_results: Dict[str, Any]) -> str:
        """Summarize tool results for context"""
        
        summary_parts = []
        
        if "items" in tool_results:
            items = tool_results["items"]
            summary_parts.append(f"Found {len(items)} items")
        
        if "translation" in tool_results:
            translation = tool_results["translation"]
            summary_parts.append(f"Translated from {translation.get('source_language')} to {translation.get('target_language')}")
        
        if "similar_items" in tool_results:
            similar = tool_results["similar_items"]
            summary_parts.append(f"Found {len(similar)} similar items")
        
        return "; ".join(summary_parts) if summary_parts else "No specific results"
    
    def _classify_response_type(self, state: DialogueState, tool_results: Dict) -> str:
        """Classify the type of response being generated"""
        
        if tool_results:
            if "items" in tool_results:
                return "product_recommendation"
            elif "translation" in tool_results:
                return "translation_response"
            else:
                return "tool_grounded"
        elif state.current_intent == "chitchat":
            return "conversational"
        else:
            return "general_assistance"

class BilingualTemplates:
    """Bilingual response templates"""
    
    def __init__(self):
        self.templates = {
            "recommendation": {
                "en": {
                    "single_item": "Perfect! I recommend {item_name} for ${price}. {description}. Would you like to see more details?",
                    "multiple_items": "Great! I found {item_count} options for you:\n{item_list}\n\nWhich one interests you most?",
                    "no_results": "I couldn't find any items matching your criteria. Let me try different search terms."
                },
                "hi": {
                    "single_item": "बेहतरीन! मैं {item_name} ${price} में सुझाता हूँ। {description}। क्या आप अधिक विवरण देखना चाहते हैं?",
                    "multiple_items": "अच्छा! मुझे आपके लिए {item_count} विकल्प मिले हैं:\n{item_list}\n\nआपको कौन सा सबसे दिलचस्प लगता है?",
                    "no_results": "मुझे आपके मापदंडों से मेल खाने वाली कोई वस्तु नहीं मिली। मैं अलग खोज शब्दों की कोशिश करता हूँ।"
                }
            },
            "general": {
                "en": {
                    "greeting": "Hello! I'm here to help you find what you're looking for. What can I assist you with today?",
                    "acknowledgment": "You're welcome! Is there anything else I can help you with?",
                    "general_help": "I'm here to help you find products and answer your questions. What would you like to explore?"
                },
                "hi": {
                    "greeting": "नमस्ते! मैं यहाँ आपको वह खोजने में मदद करने के लिए हूँ जिसकी आप तलाश कर रहे हैं। आज मैं आपकी कैसे सहायता कर सकता हूँ?",
                    "acknowledgment": "आपका स्वागत है! क्या कोई और चीज़ है जिसमें मैं आपकी मदद कर सकता हूँ?",
                    "general_help": "मैं यहाँ उत्पाद खोजने और आपके सवालों के जवाब देने के लिए हूँ। आप क्या एक्सप्लोर करना चाहते हैं?"
                }
            }
        }
    
    def get_template(self, template_type: str, language: str) -> Dict[str, str]:
        """Get template for specific type and language"""
        
        return self.templates.get(template_type, {}).get(language, 
                                                       self.templates.get(template_type, {}).get("en", {}))

# =============================================================================
# SAVE TO FILE - Part 1 Complete
# =============================================================================
