#!/usr/bin/env python3
"""
LangGraph Integration for MUSE v3
=================================

LangGraph-based conversation orchestration with state management,
conditional routing, and multimodal processing.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import json
import asyncio
import logging
import time
from enum import Enum

# LangGraph-style imports (simplified implementation)
from typing import TypedDict

logger = logging.getLogger(__name__)

# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class ConversationState(TypedDict):
    """LangGraph conversation state"""
    # User input
    user_input: str
    user_language: str
    user_profile: Dict[str, Any]
    
    # Conversation context
    conversation_history: List[Dict[str, Any]]
    current_turn: int
    session_id: str
    
    # Multimodal data
    images: Optional[List[Any]]
    visual_context: Optional[Dict[str, Any]]
    
    # Processing state
    detected_intent: Optional[str]
    extracted_entities: Dict[str, Any]
    dialogue_state: Dict[str, Any]
    
    # Tool execution
    selected_tools: List[str]
    tool_results: Dict[str, Any]
    execution_plan: Optional[Dict[str, Any]]
    
    # Response generation
    response_text: str
    response_language: str
    multimodal_elements: Dict[str, Any]
    confidence_score: float
    
    # Metadata
    processing_time: float
    error_log: List[str]
    debug_info: Dict[str, Any]

@dataclass
class ConversationFlow:
    """Define conversation flow structure"""
    
    # Core processing nodes
    PERCEPTION = "perception"
    INTENT_DETECTION = "intent_detection"  
    STATE_TRACKING = "state_tracking"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    RESPONSE_GENERATION = "response_generation"
    
    # Conditional nodes
    LANGUAGE_DETECTION = "language_detection"
    TRANSLATION_NEEDED = "translation_needed"
    MULTIMODAL_PROCESSING = "multimodal_processing"
    ERROR_RECOVERY = "error_recovery"
    
    # End states
    SUCCESS = "success"
    FAILURE = "failure"

# =============================================================================
# LANGGRAPH NODES
# =============================================================================

class LangGraphNode:
    """Base class for LangGraph nodes"""
    
    def __init__(self, name: str):
        self.name = name
        
    async def process(self, state: ConversationState) -> ConversationState:
        """Process the state and return updated state"""
        raise NotImplementedError

class PerceptionNode(LangGraphNode):
    """Process multimodal input through perception layer (real encoders)

    Uses a text encoder (Hugging Face AutoModel), a CLIP image encoder and the
    same text encoder for metadata. If encoders are passed in via constructor
    they will be used; otherwise defaults are created. All heavy calls are
    guarded with fallbacks to the original lightweight mocks.
    """

    def __init__(self, text_encoder=None, image_encoder=None, metadata_encoder=None, fusion_layer=None):
        super().__init__("perception")
        # Device
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self.device = "cpu"

        # Try to use provided encoder objects (they may already be HF models)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.metadata_encoder = metadata_encoder
        self.fusion_layer = fusion_layer

        # Lazy-init HF encoders if not provided
        try:
            if self.text_encoder is None:
                from transformers import AutoTokenizer, AutoModel
                self._hf_text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                self._hf_text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                try:
                    self._hf_text_model.to(self.device)
                except Exception:
                    pass
                self.text_encoder = "hf_text"

            if self.image_encoder is None:
                from transformers import CLIPProcessor, CLIPModel
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                try:
                    self._clip_model.to(self.device)
                except Exception:
                    pass
                self.image_encoder = "clip"

            # metadata encoder will reuse text encoder if not provided
            if self.metadata_encoder is None:
                self.metadata_encoder = self.text_encoder

            self._use_real_encoders = True
        except Exception:
            # If any import/model load fails, fall back to mock behavior
            self._use_real_encoders = False

    async def process(self, state: ConversationState) -> ConversationState:
        """Process multimodal input"""
        logger.info("Processing multimodal input through perception layer")

        try:
            # Process text
            text_result = await self._process_text(state["user_input"])

            # Process images if present
            image_result = None
            if state.get("images"):
                image_result = await self._process_images(state["images"])

            # Process metadata
            metadata_result = await self._process_metadata(state.get("user_profile", {}))

            # Fusion
            if image_result and metadata_result:
                fusion_result = await self._fuse_modalities(text_result, image_result, metadata_result)

                state["visual_context"] = {
                    "has_images": True,
                    "image_features": image_result,
                    "fusion_features": fusion_result
                }
            else:
                state["visual_context"] = {"has_images": False}

            # Update state
            state["dialogue_state"].update({
                "perception_processed": True,
                "text_features": text_result,
                "multimodal_processed": image_result is not None
            })

            # Detect language
            detected_lang = self._detect_language(state["user_input"])
            state["user_language"] = detected_lang

        except Exception as e:
            logger.error(f"Perception processing failed: {e}")
            state["error_log"].append(f"Perception error: {e}")
            # Ensure minimal visual_context fields exist to avoid downstream KeyError
            if "visual_context" not in state:
                state["visual_context"] = {"has_images": False}
            if "dialogue_state" not in state:
                state["dialogue_state"] = {}
            state["dialogue_state"].update({"perception_processed": False})
            # Keep language as auto fallback
            state["user_language"] = state.get("user_language", "auto")

        return state

    async def _process_text(self, text: str) -> Dict[str, Any]:
        """Process text input using HF AutoModel (mean pooling)."""
        if not self._use_real_encoders:
            # Fallback to mock
            return {
                "text": text,
                "embeddings": f"text_embedding_{hash(text) % 1000}",
                "processed": True
            }

        try:
            import torch
            # Tokenize
            toks = self._hf_text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            for k, v in toks.items():
                toks[k] = v.to(self.device)

            # Forward
            with torch.no_grad():
                out = self._hf_text_model(**toks)
            last_hidden = out.last_hidden_state  # (1, seq_len, dim)

            # Mean pooling (attention mask aware)
            mask = toks.get("attention_mask")
            if mask is not None:
                mask = mask.unsqueeze(-1)
                summed = (last_hidden * mask).sum(1)
                counts = mask.sum(1).clamp(min=1)
                pooled = (summed / counts).squeeze(0)
            else:
                pooled = last_hidden.mean(1).squeeze(0)

            embedding = pooled.cpu().numpy().tolist()
            return {"text": text, "embeddings": embedding, "processed": True}
        except Exception as e:
            logger.warning(f"Text encoder failed, falling back to mock: {e}")
            return {"text": text, "embeddings": f"text_embedding_{hash(text) % 1000}", "processed": True}

    async def _process_images(self, images: List[Any]) -> Dict[str, Any]:
        """Process images using CLIPModel/Processor. Accepts PIL images or file paths."""
        if not self._use_real_encoders:
            return {
                "num_images": len(images),
                "image_embeddings": [f"img_embedding_{i}" for i in range(len(images))],
                "visual_attributes": ["color", "style", "pattern"],
                "processed": True
            }

        try:
            import torch
            from PIL import Image
            processed_embeddings = []

            # Prepare images list
            pil_images = []
            for im in images:
                if isinstance(im, str):
                    try:
                        pil_images.append(Image.open(im).convert("RGB"))
                    except Exception:
                        # If path invalid, skip
                        continue
                else:
                    # Assume already a PIL image-like object
                    pil_images.append(im)

            if not pil_images:
                return {"num_images": 0, "image_embeddings": [], "visual_attributes": [], "processed": False}

            inputs = self._clip_processor(images=pil_images, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            with torch.no_grad():
                clip_out = self._clip_model.get_image_features(**{k: v for k, v in inputs.items() if k != 'pixel_values' or True})

            # If clip_out is (N, dim) or a tensor
            if hasattr(clip_out, 'cpu'):
                arr = clip_out.cpu().numpy()
                for i in range(arr.shape[0]):
                    processed_embeddings.append(arr[i].tolist())
            else:
                # Unexpected type
                processed_embeddings = [str(clip_out)]

            return {
                "num_images": len(processed_embeddings),
                "image_embeddings": processed_embeddings,
                "visual_attributes": ["color", "style", "pattern"],
                "processed": True
            }

        except Exception as e:
            logger.warning(f"Image encoder failed, falling back to mock: {e}")
            return {"num_images": len(images), "image_embeddings": [f"img_embedding_{i}" for i in range(len(images))], "visual_attributes": ["color", "style", "pattern"], "processed": True}

    async def _process_metadata(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Encode metadata by stringifying and running through the text encoder."""
        text = json.dumps(user_profile, ensure_ascii=False)
        if not self._use_real_encoders:
            return {
                "user_id": user_profile.get("user_id", "anonymous"),
                "preferences": user_profile.get("preferences", {}),
                "metadata_embeddings": f"meta_embedding_{hash(str(user_profile)) % 1000}",
                "processed": True
            }

        try:
            # Reuse text pipeline
            res = await self._process_text(text)
            return {
                "user_id": user_profile.get("user_id", "anonymous"),
                "preferences": user_profile.get("preferences", {}),
                "metadata_embeddings": res.get("embeddings", None),
                "processed": True
            }
        except Exception as e:
            logger.warning(f"Metadata encoder failed, falling back to mock: {e}")
            return {"user_id": user_profile.get("user_id", "anonymous"), "preferences": user_profile.get("preferences", {}), "metadata_embeddings": f"meta_embedding_{hash(str(user_profile)) % 1000}", "processed": True}

    async def _fuse_modalities(self, text_result: Dict, image_result: Dict, metadata_result: Dict) -> Dict[str, Any]:
        """Fuse multimodal representations into a single vector (simple weighted average)."""
        try:
            # If real encoders produced numeric embeddings, fuse them numerically
            import numpy as np

            text_emb = np.array(text_result.get("embeddings")) if isinstance(text_result.get("embeddings"), (list, tuple)) else None
            img_embs = image_result.get("image_embeddings")
            img_emb = np.array(img_embs[0]) if img_embs and isinstance(img_embs[0], (list, tuple)) else None
            meta_emb = np.array(metadata_result.get("metadata_embeddings")) if isinstance(metadata_result.get("metadata_embeddings"), (list, tuple)) else None

            parts = [p for p in [text_emb, img_emb, meta_emb] if p is not None]
            if parts:
                # Resize/trim to smallest length for naive fusion
                min_len = min(p.shape[0] for p in parts)
                parts = [p[:min_len] for p in parts]
                stacked = np.stack(parts, axis=0)
                fused = stacked.mean(axis=0)
                fused_list = fused.tolist()
                fused_id = abs(hash(str(fused_list))) % 100000
                return {
                    "fused_embedding": fused_list,
                    "attention_weights": {"text": 0.4, "image": 0.4, "metadata": 0.2},
                    "fusion_confidence": 0.9
                }
        except Exception:
            # Fall back to deterministic hash-based fusion (previous behavior)
            try:
                concat_str = str(text_result) + str(image_result) + str(metadata_result)
                fused_id = hash(concat_str) % 1000
            except Exception:
                fused_id = (hash(str(text_result)) ^ hash(str(image_result)) ^ hash(str(metadata_result))) % 1000

            return {
                "fused_embedding": f"fused_{fused_id}",
                "attention_weights": {"text": 0.4, "image": 0.4, "metadata": 0.2},
                "fusion_confidence": 0.85
            }

        # Default fallback
        return {"fused_embedding": "fused_default", "attention_weights": {"text": 0.4, "image": 0.4, "metadata": 0.2}, "fusion_confidence": 0.5}
    
    def _detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Simple heuristic detection
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        if hindi_chars > len(text) * 0.3:
            return "hi"
        elif any(word in text.lower() for word in ["namaste", "dhanyawad", "kaise", "aap"]):
            return "hi"
        else:
            return "en"

class IntentDetectionNode(LangGraphNode):
    """Detect user intent from multimodal input"""
    
    def __init__(self, intent_classifier):
        super().__init__("intent_detection")
        self.intent_classifier = intent_classifier
        
    async def process(self, state: ConversationState) -> ConversationState:
        """Detect user intent"""
        logger.info("Detecting user intent")
        
        try:
            # Extract features for intent detection
            input_features = self._extract_intent_features(state)
            
            # Classify intent
            intent_result = await self._classify_intent(input_features)
            
            state["detected_intent"] = intent_result["predicted_intent"]
            state["dialogue_state"]["intent_confidence"] = intent_result["confidence"]
            state["dialogue_state"]["intent_features"] = intent_result["features"]
            
            # Extract entities based on intent
            entities = await self._extract_entities(state["user_input"], intent_result["predicted_intent"])
            state["extracted_entities"] = entities
            
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            state["error_log"].append(f"Intent detection error: {e}")
            state["detected_intent"] = "chitchat"  # Default fallback
            
        return state
    
    def _extract_intent_features(self, state: ConversationState) -> Dict[str, Any]:
        """Extract features for intent classification"""
        return {
            "user_input": state["user_input"],
            "language": state["user_language"],
            # Safe access to visual_context to avoid KeyError
            "has_images": state.get("visual_context", {}).get("has_images", False),
            "conversation_history": state["conversation_history"],
            "user_profile": state["user_profile"]
        }
    
    async def _classify_intent(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify intent from features"""
        text = features["user_input"].lower()
        
        # Rule-based intent classification (would use actual ML model)
        if any(word in text for word in ["search", "find", "look for", "show me"]):
            intent = "search"
            confidence = 0.9
        elif any(word in text for word in ["recommend", "suggest", "what should"]):
            intent = "recommend" 
            confidence = 0.85
        elif any(word in text for word in ["compare", "difference", "vs", "versus"]):
            intent = "compare"
            confidence = 0.8
        elif any(word in text for word in ["filter", "narrow", "only show"]):
            intent = "filter"
            confidence = 0.75
        elif any(word in text for word in ["translate", "meaning", "hindi", "english"]):
            intent = "translate"
            confidence = 0.9
        elif features["has_images"]:
            intent = "visual_search"
            confidence = 0.8
        else:
            intent = "chitchat"
            confidence = 0.6
            
        return {
            "predicted_intent": intent,
            "confidence": confidence,
            "features": {"text_analyzed": True, "multimodal_context": features["has_images"]}
        }
    
    async def _extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities based on intent"""
        entities = {
            "product_mentions": [],
            "categories": [],
            "attributes": {},
            "constraints": {}
        }
        
        # Simple entity extraction (would use NER model)
        if intent == "search":
            # Extract product mentions
            products = ["laptop", "phone", "shirt", "shoes", "dress", "jacket"]
            entities["product_mentions"] = [p for p in products if p in text.lower()]
            
            # Extract categories  
            categories = ["electronics", "fashion", "home", "books"]
            entities["categories"] = [c for c in categories if c in text.lower()]
            
        elif intent == "filter":
            # Extract filter attributes
            if "under" in text or "below" in text:
                # Extract price constraints
                words = text.split()
                for i, word in enumerate(words):
                    if word.isdigit():
                        entities["constraints"]["max_price"] = int(word)
                        
        return entities

class StateTrackingNode(LangGraphNode):
    """Track dialogue state over conversation"""
    
    def __init__(self, state_tracker):
        super().__init__("state_tracking")
        self.state_tracker = state_tracker
        
    async def process(self, state: ConversationState) -> ConversationState:
        """Update dialogue state"""
        logger.info("Updating dialogue state")
        
        try:
            # Build state update input
            state_input = {
                "current_turn": state["current_turn"],
                "intent": state["detected_intent"],
                "entities": state["extracted_entities"],
                "user_input": state["user_input"],
                "conversation_history": state["conversation_history"],
                "user_profile": state["user_profile"]
            }
            
            # Update dialogue state
            updated_state = await self._update_dialogue_state(state_input)
            
            # Merge with existing state
            state["dialogue_state"].update(updated_state)
            
            # Update user profile with new information
            profile_updates = self._extract_profile_updates(state_input)
            state["user_profile"].update(profile_updates)
            
        except Exception as e:
            logger.error(f"State tracking failed: {e}")
            state["error_log"].append(f"State tracking error: {e}")
            
        return state
    
    async def _update_dialogue_state(self, state_input: Dict[str, Any]) -> Dict[str, Any]:
        """Update dialogue state components"""
        return {
            "current_goal": self._infer_goal(state_input),
            "active_constraints": self._extract_constraints(state_input),
            "user_preferences": self._update_preferences(state_input),
            "conversation_phase": self._determine_phase(state_input),
            "state_confidence": 0.8
        }
    
    def _infer_goal(self, state_input: Dict[str, Any]) -> str:
        """Infer user's goal from conversation"""
        intent = state_input["intent"]
        entities = state_input["entities"]
        
        if intent == "search" and entities["product_mentions"]:
            return f"find_{entities['product_mentions'][0]}"
        elif intent == "recommend":
            return "get_recommendations"
        elif intent == "compare":
            return "compare_products"
        else:
            return "general_assistance"
    
    def _extract_constraints(self, state_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract active constraints"""
        constraints = state_input["entities"].get("constraints", {})
        
        # Add constraints from conversation history
        history = state_input["conversation_history"]
        for turn in history[-3:]:  # Last 3 turns
            if turn["role"] == "user":
                content = turn["content"].lower()
                if "under" in content:
                    # Extract price constraint (simplified)
                    words = content.split()
                    for word in words:
                        if word.isdigit():
                            constraints["max_price"] = int(word)
                            
        return constraints
    
    def _update_preferences(self, state_input: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences"""
        current_prefs = state_input["user_profile"].get("preferences", {})
        
        # Extract preferences from current input
        user_input = state_input["user_input"].lower()
        
        # Style preferences
        if any(style in user_input for style in ["casual", "formal", "sport"]):
            for style in ["casual", "formal", "sport"]:
                if style in user_input:
                    current_prefs["style"] = style
                    
        # Color preferences
        colors = ["red", "blue", "green", "black", "white"]
        for color in colors:
            if color in user_input:
                if "colors" not in current_prefs:
                    current_prefs["colors"] = []
                if color not in current_prefs["colors"]:
                    current_prefs["colors"].append(color)
                    
        return current_prefs
    
    def _determine_phase(self, state_input: Dict[str, Any]) -> str:
        """Determine conversation phase"""
        turn_count = state_input["current_turn"]
        intent = state_input["intent"]
        
        if turn_count == 1:
            return "initial_request"
        elif intent in ["search", "recommend"]:
            return "information_gathering"
        elif intent == "compare":
            return "decision_support"
        elif intent == "chitchat":
            return "social_interaction"
        else:
            return "ongoing_assistance"
    
    def _extract_profile_updates(self, state_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract lightweight profile updates from state input (fallback implementation)."""
        # Reuse preference extraction to produce simple profile updates
        try:
            return self._update_preferences(state_input)
        except Exception:
            return {}

class ToolSelectionNode(LangGraphNode):
    """Select appropriate tools for execution"""
    
    def __init__(self, tool_selector, octotools_framework):
        super().__init__("tool_selection")
        self.tool_selector = tool_selector
        self.octotools = octotools_framework
        
    async def process(self, state: ConversationState) -> ConversationState:
        """Select tools for execution"""
        logger.info("Selecting tools for execution")
        
        try:
            # Prepare tool selection input
            selection_input = {
                "intent": state["detected_intent"],
                "entities": state["extracted_entities"],
                "dialogue_state": state["dialogue_state"],
                "language": state["user_language"],
                "multimodal_context": state["visual_context"]
            }
            
            # Select tools
            tool_selection = await self._select_tools(selection_input)
            
            state["selected_tools"] = tool_selection["tools"]
            state["execution_plan"] = tool_selection["plan"]
            
            # Log tool selection
            state["debug_info"]["tool_selection"] = {
                "selected_tools": tool_selection["tools"],
                "selection_confidence": tool_selection["confidence"],
                "reasoning": tool_selection["reasoning"]
            }
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            state["error_log"].append(f"Tool selection error: {e}")
            state["selected_tools"] = ["search"]  # Default fallback
            
        return state
    
    async def _select_tools(self, selection_input: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate tools"""
        intent = selection_input["intent"]
        entities = selection_input["entities"]
        language = selection_input["language"]
        multimodal = selection_input["multimodal_context"]
        
        selected_tools = []
        reasoning = []
        
        # Intent-based tool selection
        if intent == "search":
            if multimodal["has_images"]:
                selected_tools.append("visual_search")
                reasoning.append("Visual search selected due to image input")
            else:
                selected_tools.append("search")
                reasoning.append("Text search selected for search intent")
                
        elif intent == "recommend":
            selected_tools.append("recommend")
            reasoning.append("Recommendation tool selected for recommend intent")
            
        elif intent == "compare":
            selected_tools.extend(["search", "compare"])
            reasoning.append("Search and compare tools selected for comparison")
            
        elif intent == "filter":
            selected_tools.extend(["search", "filter"])
            reasoning.append("Search and filter tools selected for filtering")
            
        elif intent == "translate":
            selected_tools.append("translate")
            reasoning.append("Translation tool selected for language conversion")
            
        # Language-based tool addition
        if language != "en" and "translate" not in selected_tools:
            selected_tools.append("translate")
            reasoning.append("Translation tool added for non-English input")
            
        # Create execution plan
        plan = self.octotools.create_plan(
            goal=f"{intent} with tools {selected_tools}",
            context=selection_input
        )
        
        return {
            "tools": selected_tools,
            "plan": plan.__dict__,
            "confidence": 0.85,
            "reasoning": reasoning
        }

class ToolExecutionNode(LangGraphNode):
    """Execute selected tools"""
    
    def __init__(self, octotools_framework):
        super().__init__("tool_execution")
        self.octotools = octotools_framework
        
    async def process(self, state: ConversationState) -> ConversationState:
        """Execute tools"""
        logger.info(f"Executing tools: {state['selected_tools']}")
        
        try:
            # Execute tools through OctoTools framework
            execution_context = {
                "user_input": state["user_input"],
                "entities": state["extracted_entities"],
                "user_profile": state["user_profile"],
                "language": state["user_language"]
            }
            
            # Execute each tool
            results = {}
            for tool_name in state["selected_tools"]:
                tool_result = await self._execute_single_tool(tool_name, execution_context)
                results[tool_name] = tool_result
                
            state["tool_results"] = results
            
            # Update dialogue state with tool results
            state["dialogue_state"]["tools_executed"] = True
            state["dialogue_state"]["execution_success"] = all(
                result.get("success", False) for result in results.values()
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            state["error_log"].append(f"Tool execution error: {e}")
            state["tool_results"] = {"error": str(e)}
            
        return state
    
    async def _execute_single_tool(self, tool_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool"""
        tool = self.octotools.tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool {tool_name} not found"}
        
        # Prepare arguments based on tool and context
        arguments = self._prepare_tool_arguments(tool_name, context)
        
        try:
            result = await tool.execute(**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _prepare_tool_arguments(self, tool_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare arguments for tool execution"""
        entities = context["entities"]
        user_input = context["user_input"]
        user_profile = context["user_profile"]
        
        if tool_name == "search":
            return {
                "query": user_input,
                "category": entities.get("categories", ["all"])[0] if entities.get("categories") else "all",
                "max_results": 10
            }
        
        elif tool_name == "recommend":
            return {
                "user_profile": user_profile,
                "context": context,
                "num_items": 5
            }
        
        elif tool_name == "translate":
            return {
                "text": user_input,
                "source_lang": context["language"],
                "target_lang": "en" if context["language"] == "hi" else "hi"
            }
        
        elif tool_name == "visual_search":
            return {
                "image_data": "mock_image_data",  # Would use actual image
                "similarity_threshold": 0.8
            }
        
        elif tool_name == "filter":
            return {
                "items": [],  # Would come from previous search
                "filters": entities.get("constraints", {})
            }
        
        elif tool_name == "compare":
            return {
                "items": [],  # Would come from previous search
                "comparison_aspects": ["price", "rating", "features"]
            }
        
        else:
            return {}

class ResponseGenerationNode(LangGraphNode):
    """Generate final response"""
    
    def __init__(self, response_generator):
        super().__init__("response_generation")
        self.response_generator = response_generator
        
    async def process(self, state: ConversationState) -> ConversationState:
        """Generate response"""
        logger.info("Generating response")
        
        try:
            # Prepare response generation input
            generation_input = {
                "intent": state["detected_intent"],
                "tool_results": state["tool_results"],
                "dialogue_state": state["dialogue_state"],
                "user_language": state["user_language"],
                "conversation_history": state["conversation_history"]
            }
            
            # Generate response
            response = await self._generate_response(generation_input)
            
            state["response_text"] = response["text"]
            state["response_language"] = response["language"]
            state["multimodal_elements"] = response.get("multimodal_elements", {})
            state["confidence_score"] = response.get("confidence", 0.8)
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["error_log"].append(f"Response generation error: {e}")
            state["response_text"] = "I apologize, but I encountered an error. How can I help you?"
            state["response_language"] = "en"
            
        return state
    
    async def _generate_response(self, generation_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on tool results and context"""
        intent = generation_input["intent"]
        tool_results = generation_input["tool_results"]
        language = generation_input["user_language"]
        
        # Generate response based on intent and tool results
        if intent == "search":
            response_text = self._generate_search_response(tool_results)
        elif intent == "recommend":
            response_text = self._generate_recommendation_response(tool_results)
        elif intent == "compare":
            response_text = self._generate_comparison_response(tool_results)
        elif intent == "translate":
            response_text = self._generate_translation_response(tool_results)
        else:
            response_text = self._generate_generic_response(tool_results)
        
        # Translate response if needed
        if language == "hi":
            # Would use actual translation
            response_text = f"[HI] {response_text}"
        
        return {
            "text": response_text,
            "language": language,
            "confidence": 0.85,
            "multimodal_elements": self._extract_multimodal_elements(tool_results)
        }
    
    def _generate_search_response(self, tool_results: Dict[str, Any]) -> str:
        """Generate search response"""
        search_result = tool_results.get("search", {}).get("result", {})
        
        if search_result.get("success"):
            results = search_result.get("results", [])
            if results:
                return f"I found {len(results)} items for you. Here are the top results: {results[0]['title']} for ${results[0]['price']}"
            else:
                return "I didn't find any matching items. Would you like to try a different search?"
        else:
            return "I encountered an issue with the search. Could you please try again?"
    
    def _generate_recommendation_response(self, tool_results: Dict[str, Any]) -> str:
        """Generate recommendation response"""
        rec_result = tool_results.get("recommend", {}).get("result", {})
        
        if rec_result.get("success"):
            recommendations = rec_result.get("recommendations", [])
            if recommendations:
                top_rec = recommendations[0]
                return f"Based on your preferences, I recommend the {top_rec['title']} for ${top_rec['price']}. {top_rec.get('reason', '')}"
            else:
                return "I'm having trouble finding recommendations. Could you tell me more about your preferences?"
        else:
            return "I encountered an issue generating recommendations. Please try again."
    
    def _generate_comparison_response(self, tool_results: Dict[str, Any]) -> str:
        """Generate comparison response"""
        compare_result = tool_results.get("compare", {}).get("result", {})
        
        if compare_result.get("success"):
            return "I've compared the items for you. Based on the comparison, here are the key differences in price, rating, and features."
        else:
            return "I need more items to compare. Could you provide specific products you'd like me to compare?"
    
    def _generate_translation_response(self, tool_results: Dict[str, Any]) -> str:
        """Generate translation response"""
        translate_result = tool_results.get("translate", {}).get("result", {})
        
        if translate_result.get("success"):
            translated = translate_result.get("translated_text", "")
            return f"Translation: {translated}"
        else:
            return "I had trouble with the translation. Could you rephrase your request?"
    
    def _generate_generic_response(self, tool_results: Dict[str, Any]) -> str:
        """Generate generic response"""
        return "I'm here to help you find what you're looking for. What can I assist you with today?"
    
    def _extract_multimodal_elements(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract multimodal elements for response"""
        elements = {"has_images": False, "images": [], "visual_features": {}}
        
        # Check if any tool results contain visual elements
        for tool_name, tool_result in tool_results.items():
            if tool_name == "visual_search":
                result = tool_result.get("result", {})
                if result.get("success"):
                    elements["has_images"] = True
                    elements["visual_features"] = result.get("similar_items", [])
        
        return elements

# =============================================================================
# LANGGRAPH ORCHESTRATOR
# =============================================================================

class LangGraphOrchestrator:
    """Main orchestrator for LangGraph-based conversation flow"""
    
    def __init__(self, components: Dict[str, Any]):
        self.components = components
        self.conversation_flow = ConversationFlow()
        
        # Initialize nodes
        self.nodes = self._initialize_nodes()
        
        # Define flow graph
        self.graph = self._define_conversation_graph()
        
    def _initialize_nodes(self) -> Dict[str, LangGraphNode]:
        """Initialize all processing nodes"""
        return {
            self.conversation_flow.PERCEPTION: PerceptionNode(
                self.components["text_encoder"],
                self.components["image_encoder"],
                self.components["metadata_encoder"],
                self.components["fusion_layer"]
            ),
            self.conversation_flow.INTENT_DETECTION: IntentDetectionNode(
                self.components["intent_classifier"]
            ),
            self.conversation_flow.STATE_TRACKING: StateTrackingNode(
                self.components["state_tracker"]
            ),
            self.conversation_flow.TOOL_SELECTION: ToolSelectionNode(
                self.components["tool_selector"],
                self.components["octotools_framework"]
            ),
            self.conversation_flow.TOOL_EXECUTION: ToolExecutionNode(
                self.components["octotools_framework"]
            ),
            self.conversation_flow.RESPONSE_GENERATION: ResponseGenerationNode(
                self.components["response_generator"]
            )
        }
    
    def _define_conversation_graph(self) -> Dict[str, Any]:
        """Define the conversation flow graph"""
        # Simplified graph definition (would use actual LangGraph syntax)
        return {
            "start": self.conversation_flow.PERCEPTION,
            "flow": {
                self.conversation_flow.PERCEPTION: self.conversation_flow.LANGUAGE_DETECTION,
                self.conversation_flow.LANGUAGE_DETECTION: self.conversation_flow.INTENT_DETECTION,
                self.conversation_flow.INTENT_DETECTION: self.conversation_flow.STATE_TRACKING,
                self.conversation_flow.STATE_TRACKING: self.conversation_flow.TOOL_SELECTION,
                self.conversation_flow.TOOL_SELECTION: self.conversation_flow.TOOL_EXECUTION,
                self.conversation_flow.TOOL_EXECUTION: self.conversation_flow.RESPONSE_GENERATION,
                self.conversation_flow.RESPONSE_GENERATION: self.conversation_flow.SUCCESS
            },
            "conditionals": {
                self.conversation_flow.LANGUAGE_DETECTION: self._language_routing,
                self.conversation_flow.TRANSLATION_NEEDED: self._translation_routing,
                self.conversation_flow.ERROR_RECOVERY: self._error_recovery_routing
            }
        }
    
    async def process_conversation(self, user_input: str, 
                                 session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a complete conversation turn"""
        if session_context is None:
            session_context = {}
            
        # Initialize conversation state
        state = ConversationState(
            user_input=user_input,
            user_language="auto",
            user_profile=session_context.get("user_profile", {}),
            conversation_history=session_context.get("conversation_history", []),
            current_turn=session_context.get("current_turn", 1),
            session_id=session_context.get("session_id", "default"),
            images=session_context.get("images", None),
            visual_context={},
            detected_intent=None,
            extracted_entities={},
            dialogue_state={"initialized": True},
            selected_tools=[],
            tool_results={},
            execution_plan=None,
            response_text="",
            response_language="en",
            multimodal_elements={},
            confidence_score=0.0,
            processing_time=0.0,
            error_log=[],
            debug_info={}
        )
        
        # Process through conversation flow
        start_time = time.time()
        
        try:
            # Execute conversation flow
            final_state = await self._execute_flow(state)
            
            final_state["processing_time"] = time.time() - start_time
            
            # Prepare response
            response = {
                "success": True,
                "response": final_state["response_text"],
                "language": final_state["response_language"],
                "intent": final_state["detected_intent"],
                "confidence": final_state["confidence_score"],
                "multimodal_elements": final_state["multimodal_elements"],
                "processing_time": final_state["processing_time"],
                "debug_info": final_state["debug_info"] if session_context.get("debug", False) else {},
                "session_context": {
                    "conversation_history": final_state["conversation_history"] + [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": final_state["response_text"]}
                    ],
                    "current_turn": final_state["current_turn"] + 1,
                    "user_profile": final_state["user_profile"],
                    "session_id": final_state["session_id"]
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Conversation processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "processing_time": time.time() - start_time
            }
    
    async def _execute_flow(self, state: ConversationState) -> ConversationState:
        """Execute the conversation flow"""
        current_node = self.conversation_flow.PERCEPTION
        
        while current_node != self.conversation_flow.SUCCESS:
            if current_node in self.nodes:
                # Execute node
                state = await self.nodes[current_node].process(state)
                
                # Determine next node
                next_node = self._get_next_node(current_node, state)
                current_node = next_node
            else:
                # Handle conditional nodes
                current_node = await self._handle_conditional_node(current_node, state)
                
        return state
    
    def _get_next_node(self, current_node: str, state: ConversationState) -> str:
        """Get next node in flow"""
        return self.graph["flow"].get(current_node, self.conversation_flow.SUCCESS)
    
    async def _handle_conditional_node(self, node: str, state: ConversationState) -> str:
        """Handle conditional routing nodes"""
        if node == self.conversation_flow.LANGUAGE_DETECTION:
            return self._language_routing(state)
        elif node == self.conversation_flow.TRANSLATION_NEEDED:
            return self._translation_routing(state)
        else:
            return self.conversation_flow.INTENT_DETECTION
    
    def _language_routing(self, state: ConversationState) -> str:
        """Route based on detected language"""
        if state["user_language"] == "hi":
            # For Hindi input, might need translation
            return self.conversation_flow.TRANSLATION_NEEDED
        else:
            return self.conversation_flow.INTENT_DETECTION
    
    def _translation_routing(self, state: ConversationState) -> str:
        """Route based on translation needs"""
        # For now, proceed to intent detection
        return self.conversation_flow.INTENT_DETECTION
    
    def _error_recovery_routing(self, state: ConversationState) -> str:
        """Route based on error recovery"""
        if state["error_log"]:
            # Try recovery or fallback
            return self.conversation_flow.RESPONSE_GENERATION
        else:
            return self.conversation_flow.SUCCESS
