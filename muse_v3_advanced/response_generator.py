#!/usr/bin/env python3
"""
MUSE v3 Response Generator
=========================

Bilingual response generation with Hindi-English support,
template-based responses, and multimodal output capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import re
import logging
from dataclasses import dataclass
from enum import Enum
import random
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# RESPONSE GENERATION CONFIG
# =============================================================================

class ResponseType(Enum):
    """Types of responses"""
    SEARCH_RESULTS = "search_results"
    RECOMMENDATIONS = "recommendations"  
    COMPARISON = "comparison"
    FILTER_RESULTS = "filter_results"
    TRANSLATION = "translation"
    VISUAL_SEARCH = "visual_search"
    CHITCHAT = "chitchat"
    ERROR = "error"
    CLARIFICATION = "clarification"

@dataclass
class ResponseTemplate:
    """Response template structure"""
    template_id: str
    response_type: ResponseType
    language: str
    template_text: str
    required_slots: List[str]
    optional_slots: List[str]
    multimodal_elements: List[str]
    tone: str  # formal, casual, friendly, professional

@dataclass  
class GenerationContext:
    """Context for response generation"""
    intent: str
    tool_results: Dict[str, Any]
    user_language: str
    conversation_history: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    dialogue_state: Dict[str, Any]
    multimodal_context: Dict[str, Any]
    confidence_score: float

# =============================================================================
# BILINGUAL TEMPLATES
# =============================================================================

class BilingualTemplateManager:
    """Manages bilingual response templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, Dict[str, List[ResponseTemplate]]]:
        """Initialize bilingual response templates"""
        templates = {
            "en": {
                ResponseType.SEARCH_RESULTS.value: self._get_search_templates_en(),
                ResponseType.RECOMMENDATIONS.value: self._get_recommendation_templates_en(),
                ResponseType.COMPARISON.value: self._get_comparison_templates_en(),
                ResponseType.FILTER_RESULTS.value: self._get_filter_templates_en(),
                ResponseType.TRANSLATION.value: self._get_translation_templates_en(),
                ResponseType.VISUAL_SEARCH.value: self._get_visual_search_templates_en(),
                ResponseType.CHITCHAT.value: self._get_chitchat_templates_en(),
                ResponseType.ERROR.value: self._get_error_templates_en(),
                ResponseType.CLARIFICATION.value: self._get_clarification_templates_en()
            },
            "hi": {
                ResponseType.SEARCH_RESULTS.value: self._get_search_templates_hi(),
                ResponseType.RECOMMENDATIONS.value: self._get_recommendation_templates_hi(),
                ResponseType.COMPARISON.value: self._get_comparison_templates_hi(),
                ResponseType.FILTER_RESULTS.value: self._get_filter_templates_hi(),
                ResponseType.TRANSLATION.value: self._get_translation_templates_hi(),
                ResponseType.VISUAL_SEARCH.value: self._get_visual_search_templates_hi(),
                ResponseType.CHITCHAT.value: self._get_chitchat_templates_hi(),
                ResponseType.ERROR.value: self._get_error_templates_hi(),
                ResponseType.CLARIFICATION.value: self._get_clarification_templates_hi()
            }
        }
        return templates
    
    def _get_search_templates_en(self) -> List[ResponseTemplate]:
        """English search result templates"""
        return [
            ResponseTemplate(
                template_id="search_success_en_1",
                response_type=ResponseType.SEARCH_RESULTS,
                language="en",
                template_text="I found {num_results} items matching your search for '{query}'. Here are the top results:",
                required_slots=["num_results", "query"],
                optional_slots=["category", "price_range"],
                multimodal_elements=["product_images", "price_display"],
                tone="friendly"
            ),
            ResponseTemplate(
                template_id="search_success_en_2",
                response_type=ResponseType.SEARCH_RESULTS,
                language="en",
                template_text="Great! I discovered {num_results} {category} items for you. The top result is {top_item} for ${price}.",
                required_slots=["num_results", "top_item", "price"],
                optional_slots=["category", "rating"],
                multimodal_elements=["featured_image", "price_tag"],
                tone="enthusiastic"
            ),
            ResponseTemplate(
                template_id="search_no_results_en",
                response_type=ResponseType.SEARCH_RESULTS,
                language="en",
                template_text="I couldn't find any items matching '{query}'. Would you like to try a different search term or browse {suggested_category}?",
                required_slots=["query"],
                optional_slots=["suggested_category"],
                multimodal_elements=["suggestion_images"],
                tone="helpful"
            )
        ]
    
    def _get_search_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi search result templates"""
        return [
            ResponseTemplate(
                template_id="search_success_hi_1",
                response_type=ResponseType.SEARCH_RESULTS,
                language="hi",
                template_text="मुझे आपकी खोज '{query}' के लिए {num_results} आइटम मिले हैं। यहाँ सबसे अच्छे परिणाम हैं:",
                required_slots=["num_results", "query"],
                optional_slots=["category", "price_range"],
                multimodal_elements=["product_images", "price_display"],
                tone="friendly"
            ),
            ResponseTemplate(
                template_id="search_success_hi_2",
                response_type=ResponseType.SEARCH_RESULTS,
                language="hi",
                template_text="बहुत अच्छा! मुझे आपके लिए {num_results} {category} आइटम मिले हैं। सबसे अच्छा विकल्प {top_item} है जो ${price} में है।",
                required_slots=["num_results", "top_item", "price"],
                optional_slots=["category", "rating"],
                multimodal_elements=["featured_image", "price_tag"],
                tone="enthusiastic"
            ),
            ResponseTemplate(
                template_id="search_no_results_hi",
                response_type=ResponseType.SEARCH_RESULTS,
                language="hi",
                template_text="मुझे '{query}' से मेल खाने वाले कोई आइटम नहीं मिले। क्या आप कोई अलग खोज शब्द आज़माना चाहेंगे या {suggested_category} देखना चाहेंगे?",
                required_slots=["query"],
                optional_slots=["suggested_category"],
                multimodal_elements=["suggestion_images"],
                tone="helpful"
            )
        ]
    
    def _get_recommendation_templates_en(self) -> List[ResponseTemplate]:
        """English recommendation templates"""
        return [
            ResponseTemplate(
                template_id="rec_personal_en_1",
                response_type=ResponseType.RECOMMENDATIONS,
                language="en",
                template_text="Based on your preferences for {preference_type}, I recommend the {item_name}. It's {rating} rated and costs ${price}. {reason}",
                required_slots=["item_name", "price", "reason"],
                optional_slots=["preference_type", "rating"],
                multimodal_elements=["recommendation_card", "rating_stars"],
                tone="personalized"
            ),
            ResponseTemplate(
                template_id="rec_trending_en",
                response_type=ResponseType.RECOMMENDATIONS,
                language="en",
                template_text="Here are {num_items} trending items that match your style: {item_list}. These are popular right now!",
                required_slots=["num_items", "item_list"],
                optional_slots=["style_preference"],
                multimodal_elements=["trending_badge", "item_grid"],
                tone="trendy"
            )
        ]
    
    def _get_recommendation_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi recommendation templates"""
        return [
            ResponseTemplate(
                template_id="rec_personal_hi_1",
                response_type=ResponseType.RECOMMENDATIONS,
                language="hi",
                template_text="आपकी {preference_type} की पसंद के आधार पर, मैं {item_name} की सिफारिश करता हूँ। इसकी रेटिंग {rating} है और कीमत ${price} है। {reason}",
                required_slots=["item_name", "price", "reason"],
                optional_slots=["preference_type", "rating"],
                multimodal_elements=["recommendation_card", "rating_stars"],
                tone="personalized"
            ),
            ResponseTemplate(
                template_id="rec_trending_hi",
                response_type=ResponseType.RECOMMENDATIONS,
                language="hi",
                template_text="यहाँ {num_items} ट्रेंडिंग आइटम हैं जो आपकी स्टाइल से मेल खाते हैं: {item_list}। ये अभी बहुत लोकप्रिय हैं!",
                required_slots=["num_items", "item_list"],
                optional_slots=["style_preference"],
                multimodal_elements=["trending_badge", "item_grid"],
                tone="trendy"
            )
        ]
    
    def _get_comparison_templates_en(self) -> List[ResponseTemplate]:
        """English comparison templates"""
        return [
            ResponseTemplate(
                template_id="comparison_detailed_en",
                response_type=ResponseType.COMPARISON,
                language="en",
                template_text="Here's how {item1} compares to {item2}: {item1} costs ${price1} with {rating1} rating, while {item2} costs ${price2} with {rating2} rating. {recommendation}",
                required_slots=["item1", "item2", "price1", "price2", "recommendation"],
                optional_slots=["rating1", "rating2", "features"],
                multimodal_elements=["comparison_table", "vs_graphic"],
                tone="analytical"
            ),
            ResponseTemplate(
                template_id="comparison_winner_en",
                response_type=ResponseType.COMPARISON,
                language="en",
                template_text="Between {item1} and {item2}, I'd recommend {winner} because {reason}. It offers better {advantage}.",
                required_slots=["item1", "item2", "winner", "reason"],
                optional_slots=["advantage"],
                multimodal_elements=["winner_highlight", "advantage_icons"],
                tone="decisive"
            )
        ]
    
    def _get_comparison_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi comparison templates"""
        return [
            ResponseTemplate(
                template_id="comparison_detailed_hi",
                response_type=ResponseType.COMPARISON,
                language="hi",
                template_text="यहाँ देखें कि {item1} और {item2} में कैसा तुलना है: {item1} की कीमत ${price1} है और रेटिंग {rating1} है, जबकि {item2} की कीमत ${price2} है और रेटिंग {rating2} है। {recommendation}",
                required_slots=["item1", "item2", "price1", "price2", "recommendation"],
                optional_slots=["rating1", "rating2", "features"],
                multimodal_elements=["comparison_table", "vs_graphic"],
                tone="analytical"
            ),
            ResponseTemplate(
                template_id="comparison_winner_hi",
                response_type=ResponseType.COMPARISON,
                language="hi",
                template_text="{item1} और {item2} के बीच, मैं {winner} की सिफारिश करूंगा क्योंकि {reason}। यह बेहतर {advantage} प्रदान करता है।",
                required_slots=["item1", "item2", "winner", "reason"],
                optional_slots=["advantage"],
                multimodal_elements=["winner_highlight", "advantage_icons"],
                tone="decisive"
            )
        ]
    
    def _get_filter_templates_en(self) -> List[ResponseTemplate]:
        """English filter templates"""
        return [
            ResponseTemplate(
                template_id="filter_applied_en",
                response_type=ResponseType.FILTER_RESULTS,
                language="en",
                template_text="I've filtered the results based on your criteria: {filter_criteria}. Found {num_results} items {price_range}.",
                required_slots=["filter_criteria", "num_results"],
                optional_slots=["price_range", "category"],
                multimodal_elements=["filter_tags", "result_grid"],
                tone="efficient"
            )
        ]
    
    def _get_filter_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi filter templates"""
        return [
            ResponseTemplate(
                template_id="filter_applied_hi",
                response_type=ResponseType.FILTER_RESULTS,
                language="hi",
                template_text="मैंने आपकी शर्तों के आधार पर परिणामों को फ़िल्टर किया है: {filter_criteria}। {price_range} में {num_results} आइटम मिले हैं।",
                required_slots=["filter_criteria", "num_results"],
                optional_slots=["price_range", "category"],
                multimodal_elements=["filter_tags", "result_grid"],
                tone="efficient"
            )
        ]
    
    def _get_translation_templates_en(self) -> List[ResponseTemplate]:
        """English translation templates"""
        return [
            ResponseTemplate(
                template_id="translation_en",
                response_type=ResponseType.TRANSLATION,
                language="en",
                template_text="Translation: '{translated_text}' ({source_lang} to {target_lang})",
                required_slots=["translated_text", "source_lang", "target_lang"],
                optional_slots=["confidence"],
                multimodal_elements=["language_flags"],
                tone="informative"
            )
        ]
    
    def _get_translation_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi translation templates"""
        return [
            ResponseTemplate(
                template_id="translation_hi",
                response_type=ResponseType.TRANSLATION,
                language="hi",
                template_text="अनुवाद: '{translated_text}' ({source_lang} से {target_lang})",
                required_slots=["translated_text", "source_lang", "target_lang"],
                optional_slots=["confidence"],
                multimodal_elements=["language_flags"],
                tone="informative"
            )
        ]
    
    def _get_visual_search_templates_en(self) -> List[ResponseTemplate]:
        """English visual search templates"""
        return [
            ResponseTemplate(
                template_id="visual_search_success_en",
                response_type=ResponseType.VISUAL_SEARCH,
                language="en",
                template_text="I found {num_similar} items similar to your image. The closest match is {top_match} with {similarity}% similarity.",
                required_slots=["num_similar", "top_match"],
                optional_slots=["similarity"],
                multimodal_elements=["similarity_comparison", "match_grid"],
                tone="technical"
            )
        ]
    
    def _get_visual_search_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi visual search templates"""
        return [
            ResponseTemplate(
                template_id="visual_search_success_hi",
                response_type=ResponseType.VISUAL_SEARCH,
                language="hi",
                template_text="मुझे आपकी तस्वीर के समान {num_similar} आइटम मिले हैं। सबसे करीबी मैच {top_match} है {similarity}% समानता के साथ।",
                required_slots=["num_similar", "top_match"],
                optional_slots=["similarity"],
                multimodal_elements=["similarity_comparison", "match_grid"],
                tone="technical"
            )
        ]
    
    def _get_chitchat_templates_en(self) -> List[ResponseTemplate]:
        """English chitchat templates"""
        return [
            ResponseTemplate(
                template_id="greeting_en",
                response_type=ResponseType.CHITCHAT,
                language="en",
                template_text="Hello! I'm here to help you find what you're looking for. What can I assist you with today?",
                required_slots=[],
                optional_slots=["user_name"],
                multimodal_elements=["welcome_icon"],
                tone="friendly"
            ),
            ResponseTemplate(
                template_id="how_can_help_en",
                response_type=ResponseType.CHITCHAT,
                language="en",
                template_text="I can help you search for products, get recommendations, compare items, or answer questions. What would you like to do?",
                required_slots=[],
                optional_slots=[],
                multimodal_elements=["feature_icons"],
                tone="helpful"
            )
        ]
    
    def _get_chitchat_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi chitchat templates"""
        return [
            ResponseTemplate(
                template_id="greeting_hi",
                response_type=ResponseType.CHITCHAT,
                language="hi",
                template_text="नमस्ते! मैं आपको जो आप खोज रहे हैं उसे खोजने में मदद करने के लिए यहाँ हूँ। आज मैं आपकी कैसे सहायता कर सकता हूँ?",
                required_slots=[],
                optional_slots=["user_name"],
                multimodal_elements=["welcome_icon"],
                tone="friendly"
            ),
            ResponseTemplate(
                template_id="how_can_help_hi",
                response_type=ResponseType.CHITCHAT,
                language="hi",
                template_text="मैं आपको उत्पादों की खोज करने, सिफारिशें पाने, आइटमों की तुलना करने या सवालों के जवाब देने में मदद कर सकता हूँ। आप क्या करना चाहेंगे?",
                required_slots=[],
                optional_slots=[],
                multimodal_elements=["feature_icons"],
                tone="helpful"
            )
        ]
    
    def _get_error_templates_en(self) -> List[ResponseTemplate]:
        """English error templates"""
        return [
            ResponseTemplate(
                template_id="general_error_en",
                response_type=ResponseType.ERROR,
                language="en",
                template_text="I apologize, but I encountered an issue: {error_message}. Please try again or rephrase your request.",
                required_slots=["error_message"],
                optional_slots=["suggestion"],
                multimodal_elements=["error_icon"],
                tone="apologetic"
            )
        ]
    
    def _get_error_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi error templates"""
        return [
            ResponseTemplate(
                template_id="general_error_hi",
                response_type=ResponseType.ERROR,
                language="hi",
                template_text="मुझे खुशी है, लेकिन मुझे एक समस्या का सामना करना पड़ा: {error_message}। कृपया फिर से कोशिश करें या अपना अनुरोध दूसरे तरीके से रखें।",
                required_slots=["error_message"],
                optional_slots=["suggestion"],
                multimodal_elements=["error_icon"],
                tone="apologetic"
            )
        ]
    
    def _get_clarification_templates_en(self) -> List[ResponseTemplate]:
        """English clarification templates"""
        return [
            ResponseTemplate(
                template_id="clarify_search_en",
                response_type=ResponseType.CLARIFICATION,
                language="en",
                template_text="I want to make sure I understand correctly. Are you looking for {clarification_item} in the {category} category?",
                required_slots=["clarification_item"],
                optional_slots=["category"],
                multimodal_elements=["question_icon"],
                tone="clarifying"
            )
        ]
    
    def _get_clarification_templates_hi(self) -> List[ResponseTemplate]:
        """Hindi clarification templates"""
        return [
            ResponseTemplate(
                template_id="clarify_search_hi",
                response_type=ResponseType.CLARIFICATION,
                language="hi",
                template_text="मैं यह सुनिश्चित करना चाहता हूँ कि मैं सही तरीके से समझ गया हूँ। क्या आप {category} श्रेणी में {clarification_item} की तलाश कर रहे हैं?",
                required_slots=["clarification_item"],
                optional_slots=["category"],
                multimodal_elements=["question_icon"],
                tone="clarifying"
            )
        ]
    
    def get_template(self, response_type: ResponseType, language: str, 
                    context: Dict[str, Any] = None) -> ResponseTemplate:
        """Get appropriate template for response type and language"""
        language = language if language in ["en", "hi"] else "en"
        
        available_templates = self.templates[language].get(response_type.value, [])
        if not available_templates:
            # Fallback to English if Hindi template not available
            available_templates = self.templates["en"].get(response_type.value, [])
        
        if not available_templates:
            # Ultimate fallback
            return self._get_fallback_template(language)
        
        # Select template based on context
        return self._select_best_template(available_templates, context)
    
    def _select_best_template(self, templates: List[ResponseTemplate], 
                            context: Dict[str, Any] = None) -> ResponseTemplate:
        """Select best template based on context"""
        if not context:
            return random.choice(templates)
        
        # Score templates based on context
        best_template = templates[0]
        best_score = 0
        
        for template in templates:
            score = self._score_template(template, context)
            if score > best_score:
                best_score = score
                best_template = template
        
        return best_template
    
    def _score_template(self, template: ResponseTemplate, context: Dict[str, Any]) -> float:
        """Score template relevance to context"""
        score = 0.0
        
        # Check if required slots are available in context
        required_available = sum(1 for slot in template.required_slots if slot in context)
        if template.required_slots:
            score += (required_available / len(template.required_slots)) * 0.6
        
        # Check optional slots
        optional_available = sum(1 for slot in template.optional_slots if slot in context)
        if template.optional_slots:
            score += (optional_available / len(template.optional_slots)) * 0.3
        
        # Tone preference (could be enhanced with user profile)
        if context.get("preferred_tone") == template.tone:
            score += 0.1
        
        return score
    
    def _get_fallback_template(self, language: str) -> ResponseTemplate:
        """Get fallback template when no specific template found"""
        if language == "hi":
            return ResponseTemplate(
                template_id="fallback_hi",
                response_type=ResponseType.CHITCHAT,
                language="hi",
                template_text="मैं आपकी सहायता करने की कोशिश कर रहा हूँ। कृपया अपना अनुरोध दूसरे तरीके से व्यक्त करें।",
                required_slots=[],
                optional_slots=[],
                multimodal_elements=[],
                tone="helpful"
            )
        else:
            return ResponseTemplate(
                template_id="fallback_en",
                response_type=ResponseType.CHITCHAT,
                language="en",
                template_text="I'm here to help you. Could you please rephrase your request?",
                required_slots=[],
                optional_slots=[],
                multimodal_elements=[],
                tone="helpful"
            )

# =============================================================================
# NEURAL RESPONSE GENERATOR
# =============================================================================

class NeuralResponseGenerator(nn.Module):
    """Neural network for generating personalized responses"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Response adaptation network
        self.context_encoder = nn.Linear(config["context_dim"], config["hidden_dim"])
        self.user_adapter = nn.Linear(config["user_dim"], config["hidden_dim"])
        self.fusion_layer = nn.Linear(config["hidden_dim"] * 2, config["hidden_dim"])
        
        # Response style classifier
        self.style_classifier = nn.Linear(config["hidden_dim"], config["num_styles"])
        
        # Response length predictor
        self.length_predictor = nn.Linear(config["hidden_dim"], 1)
        
        # Confidence estimator
        self.confidence_estimator = nn.Linear(config["hidden_dim"], 1)
        
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        
    def forward(self, context_features: torch.Tensor, 
                user_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate response adaptations"""
        # Encode context and user
        context_encoded = self.activation(self.context_encoder(context_features))
        user_encoded = self.activation(self.user_adapter(user_features))
        
        # Fuse representations
        fused = torch.cat([context_encoded, user_encoded], dim=-1)
        fused = self.activation(self.fusion_layer(fused))
        fused = self.dropout(fused)
        
        # Predict response characteristics
        style_logits = self.style_classifier(fused)
        length_prediction = torch.sigmoid(self.length_predictor(fused))
        confidence = torch.sigmoid(self.confidence_estimator(fused))
        
        return {
            "style_logits": style_logits,
            "predicted_length": length_prediction,
            "confidence": confidence,
            "response_features": fused
        }

# =============================================================================
# MAIN RESPONSE GENERATOR
# =============================================================================

class MuseV3ResponseGenerator:
    """Main response generator for MUSE v3"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        self.template_manager = BilingualTemplateManager()
        
        # Initialize neural components if available
        try:
            self.neural_generator = NeuralResponseGenerator(config["neural"])
            self.use_neural = True
        except Exception as e:
            logger.warning(f"Neural generator not available: {e}")
            self.use_neural = False
        
        # Response generation statistics
        self.stats = {
            "total_responses": 0,
            "by_language": {"en": 0, "hi": 0},
            "by_type": {},
            "average_confidence": 0.0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "neural": {
                "context_dim": 512,
                "user_dim": 256,
                "hidden_dim": 256,
                "num_styles": 4,  # formal, casual, friendly, professional
            },
            "generation": {
                "max_length": 200,
                "min_confidence": 0.6,
                "enable_personalization": True,
                "use_multimodal": True
            },
            "language": {
                "default_language": "en",
                "auto_translate": True,
                "detect_language": True
            }
        }
    
    def generate_response(self, context: GenerationContext) -> Dict[str, Any]:
        """Generate response for given context"""
        logger.info(f"Generating response for intent: {context.intent}, language: {context.user_language}")
        
        try:
            # Determine response type from intent and tool results
            response_type = self._determine_response_type(context)
            
            # Extract slots from tool results and context
            slots = self._extract_slots(context, response_type)
            
            # Get appropriate template
            template = self.template_manager.get_template(response_type, context.user_language, slots)
            
            # Generate response text
            response_text = self._fill_template(template, slots)
            
            # Add personalization if neural generator available
            if self.use_neural and self.config["generation"]["enable_personalization"]:
                response_text = self._personalize_response(response_text, context, template)
            
            # Extract multimodal elements
            multimodal_elements = self._extract_multimodal_elements(context, template)
            
            # Calculate confidence
            confidence = self._calculate_confidence(context, template, slots)
            
            # Update statistics
            self._update_stats(context.user_language, response_type, confidence)
            
            # Build response
            response = {
                "text": response_text,
                "language": template.language,
                "response_type": response_type.value,
                "template_id": template.template_id,
                "confidence": confidence,
                "multimodal_elements": multimodal_elements,
                "tone": template.tone,
                "metadata": {
                    "slots_filled": len([s for s in template.required_slots if s in slots]),
                    "total_slots": len(template.required_slots),
                    "personalized": self.use_neural,
                    "generation_time": 0.1  # Would track actual time
                }
            }
            
            logger.info(f"Generated response: {response['text'][:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_error_response(context, str(e))
    
    def _determine_response_type(self, context: GenerationContext) -> ResponseType:
        """Determine response type from context"""
        intent = context.intent
        tool_results = context.tool_results
        
        # Check if there are errors
        if any(not result.get("success", True) for result in tool_results.values()):
            return ResponseType.ERROR
        
        # Map intent to response type
        intent_mapping = {
            "search": ResponseType.SEARCH_RESULTS,
            "recommend": ResponseType.RECOMMENDATIONS,
            "compare": ResponseType.COMPARISON,
            "filter": ResponseType.FILTER_RESULTS,
            "translate": ResponseType.TRANSLATION,
            "visual_search": ResponseType.VISUAL_SEARCH,
            "chitchat": ResponseType.CHITCHAT
        }
        
        return intent_mapping.get(intent, ResponseType.CHITCHAT)
    
    def _extract_slots(self, context: GenerationContext, 
                      response_type: ResponseType) -> Dict[str, Any]:
        """Extract slots from context for template filling"""
        slots = {}
        tool_results = context.tool_results
        
        if response_type == ResponseType.SEARCH_RESULTS:
            search_result = tool_results.get("search", {}).get("result", {})
            if search_result:
                results = search_result.get("results", [])
                slots.update({
                    "num_results": len(results),
                    "query": context.dialogue_state.get("last_query", "your search"),
                    "category": search_result.get("category", "items")
                })
                
                if results:
                    top_result = results[0]
                    slots.update({
                        "top_item": top_result.get("title", "item"),
                        "price": top_result.get("price", 0),
                        "rating": top_result.get("rating", 0)
                    })
        
        elif response_type == ResponseType.RECOMMENDATIONS:
            rec_result = tool_results.get("recommend", {}).get("result", {})
            if rec_result:
                recommendations = rec_result.get("recommendations", [])
                if recommendations:
                    top_rec = recommendations[0]
                    slots.update({
                        "item_name": top_rec.get("title", "recommended item"),
                        "price": top_rec.get("price", 0),
                        "rating": top_rec.get("rating", 0),
                        "reason": top_rec.get("reason", "it matches your preferences"),
                        "num_items": len(recommendations)
                    })
                    
                    # Extract item list for multiple recommendations
                    item_names = [item.get("title", f"Item {i+1}") 
                                for i, item in enumerate(recommendations[:3])]
                    slots["item_list"] = ", ".join(item_names)
        
        elif response_type == ResponseType.COMPARISON:
            compare_result = tool_results.get("compare", {}).get("result", {})
            if compare_result:
                items = compare_result.get("comparison_items", [])
                if len(items) >= 2:
                    slots.update({
                        "item1": items[0].get("title", "Item 1"),
                        "item2": items[1].get("title", "Item 2"),
                        "price1": items[0].get("price", 0),
                        "price2": items[1].get("price", 0),
                        "rating1": items[0].get("rating", 0),
                        "rating2": items[1].get("rating", 0),
                        "winner": compare_result.get("recommendation", items[0].get("title", "first item")),
                        "reason": compare_result.get("reason", "better value"),
                        "recommendation": compare_result.get("summary", "both are good options")
                    })
        
        elif response_type == ResponseType.TRANSLATION:
            translate_result = tool_results.get("translate", {}).get("result", {})
            if translate_result:
                slots.update({
                    "translated_text": translate_result.get("translated_text", "translation not available"),
                    "source_lang": translate_result.get("source_language", "unknown"),
                    "target_lang": translate_result.get("target_language", "unknown"),
                    "confidence": translate_result.get("confidence", 0.8)
                })
        
        elif response_type == ResponseType.VISUAL_SEARCH:
            visual_result = tool_results.get("visual_search", {}).get("result", {})
            if visual_result:
                similar_items = visual_result.get("similar_items", [])
                slots.update({
                    "num_similar": len(similar_items),
                    "similarity": visual_result.get("max_similarity", 85)
                })
                
                if similar_items:
                    slots["top_match"] = similar_items[0].get("title", "similar item")
        
        elif response_type == ResponseType.FILTER_RESULTS:
            filter_result = tool_results.get("filter", {}).get("result", {})
            if filter_result:
                filtered_items = filter_result.get("filtered_items", [])
                applied_filters = filter_result.get("applied_filters", {})
                
                slots.update({
                    "num_results": len(filtered_items),
                    "filter_criteria": self._format_filter_criteria(applied_filters)
                })
                
                if "price" in applied_filters:
                    price_range = applied_filters["price"]
                    slots["price_range"] = f"between ${price_range.get('min', 0)} and ${price_range.get('max', 1000)}"
        
        # Add user profile information
        user_profile = context.user_profile
        if user_profile:
            preferences = user_profile.get("preferences", {})
            slots.update({
                "user_name": user_profile.get("name", ""),
                "preference_type": preferences.get("style", "your preferences"),
                "style_preference": preferences.get("style", ""),
                "preferred_categories": ", ".join(preferences.get("categories", []))
            })
        
        # Add error information if present
        if response_type == ResponseType.ERROR:
            error_messages = []
            for tool_name, result in tool_results.items():
                if not result.get("success", True):
                    error_messages.append(result.get("error", f"{tool_name} failed"))
            
            slots["error_message"] = "; ".join(error_messages) if error_messages else "unknown error"
        
        return slots
    
    def _format_filter_criteria(self, filters: Dict[str, Any]) -> str:
        """Format filter criteria for display"""
        criteria_parts = []
        
        for key, value in filters.items():
            if key == "price":
                if isinstance(value, dict):
                    if "max" in value:
                        criteria_parts.append(f"price under ${value['max']}")
                    elif "min" in value:
                        criteria_parts.append(f"price above ${value['min']}")
                    else:
                        criteria_parts.append(f"price range ${value.get('min', 0)}-${value.get('max', 1000)}")
            elif key == "category":
                criteria_parts.append(f"category: {value}")
            elif key == "brand":
                criteria_parts.append(f"brand: {value}")
            elif key == "rating":
                criteria_parts.append(f"rating above {value}")
            else:
                criteria_parts.append(f"{key}: {value}")
        
        return ", ".join(criteria_parts) if criteria_parts else "your preferences"
    
    def _fill_template(self, template: ResponseTemplate, slots: Dict[str, Any]) -> str:
        """Fill template with extracted slots"""
        response_text = template.template_text
        
        # Fill required slots
        for slot in template.required_slots:
            placeholder = f"{{{slot}}}"
            if slot in slots:
                value = slots[slot]
                # Format value based on type
                if isinstance(value, float):
                    if slot.endswith("_price") or slot == "price" or slot.startswith("price"):
                        value = f"{value:.2f}"
                    else:
                        value = f"{value:.1f}"
                elif isinstance(value, int) and slot not in ["num_results", "num_items", "num_similar"]:
                    value = str(value)
                
                response_text = response_text.replace(placeholder, str(value))
            else:
                # Handle missing required slots
                response_text = response_text.replace(placeholder, f"[{slot}]")
        
        # Fill optional slots
        for slot in template.optional_slots:
            placeholder = f"{{{slot}}}"
            if slot in slots and placeholder in response_text:
                value = slots[slot]
                if isinstance(value, float):
                    if slot.endswith("_price") or slot == "price":
                        value = f"{value:.2f}"
                    else:
                        value = f"{value:.1f}"
                response_text = response_text.replace(placeholder, str(value))
            else:
                # Remove optional placeholders that weren't filled
                response_text = re.sub(rf'\s*\{{{slot}\}}', '', response_text)
        
        # Clean up extra spaces
        response_text = re.sub(r'\s+', ' ', response_text).strip()
        
        return response_text
    
    def _personalize_response(self, response_text: str, context: GenerationContext, 
                            template: ResponseTemplate) -> str:
        """Personalize response using neural generator"""
        # This would use the neural generator for personalization
        # For now, apply simple rule-based personalization
        
        user_profile = context.user_profile
        if not user_profile:
            return response_text
        
        # Add personalization based on user history
        history = user_profile.get("history", {})
        if history.get("purchases", 0) > 10:
            # Experienced user - more direct
            response_text = response_text.replace("I recommend", "You might like")
        elif history.get("purchases", 0) == 0:
            # New user - more explanatory
            response_text += " Let me know if you need more details!"
        
        # Add user name if available
        user_name = user_profile.get("name", "")
        if user_name and template.tone == "friendly":
            response_text = f"{user_name}, {response_text.lower()}"
        
        return response_text
    
    def _extract_multimodal_elements(self, context: GenerationContext, 
                                   template: ResponseTemplate) -> Dict[str, Any]:
        """Extract multimodal elements for response"""
        elements = {
            "has_visual": len(template.multimodal_elements) > 0,
            "elements": []
        }
        
        tool_results = context.tool_results
        
        for element_type in template.multimodal_elements:
            element = {"type": element_type, "data": {}}
            
            if element_type == "product_images":
                # Extract product images from search results
                search_result = tool_results.get("search", {}).get("result", {})
                results = search_result.get("results", [])
                element["data"]["images"] = [
                    {"url": item.get("image_url", ""), "title": item.get("title", "")}
                    for item in results[:3]
                ]
            
            elif element_type == "comparison_table":
                # Extract comparison data
                compare_result = tool_results.get("compare", {}).get("result", {})
                element["data"]["comparison"] = compare_result.get("comparison_table", {})
            
            elif element_type == "rating_stars":
                # Add rating visualization
                element["data"]["rating"] = context.tool_results.get("search", {}).get("result", {}).get("results", [{}])[0].get("rating", 0)
            
            elif element_type in ["welcome_icon", "error_icon", "question_icon"]:
                # Static icons
                element["data"]["icon_type"] = element_type.replace("_icon", "")
            
            elements["elements"].append(element)
        
        return elements
    
    def _calculate_confidence(self, context: GenerationContext, 
                            template: ResponseTemplate, slots: Dict[str, Any]) -> float:
        """Calculate response confidence"""
        confidence = context.confidence_score
        
        # Adjust based on template slot filling
        required_filled = sum(1 for slot in template.required_slots if slot in slots)
        if template.required_slots:
            slot_completeness = required_filled / len(template.required_slots)
            confidence *= slot_completeness
        
        # Adjust based on tool success
        tool_success_rate = sum(1 for result in context.tool_results.values() 
                              if result.get("success", False)) / max(len(context.tool_results), 1)
        confidence *= (0.5 + 0.5 * tool_success_rate)
        
        # Language confidence
        if context.user_language == template.language:
            confidence *= 1.0  # Perfect match
        elif template.language == "en":
            confidence *= 0.9  # Good fallback
        else:
            confidence *= 0.8  # Translation needed
        
        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def _update_stats(self, language: str, response_type: ResponseType, confidence: float):
        """Update generation statistics"""
        self.stats["total_responses"] += 1
        self.stats["by_language"][language] = self.stats["by_language"].get(language, 0) + 1
        self.stats["by_type"][response_type.value] = self.stats["by_type"].get(response_type.value, 0) + 1
        
        # Update average confidence
        total = self.stats["total_responses"]
        prev_avg = self.stats["average_confidence"]
        self.stats["average_confidence"] = (prev_avg * (total - 1) + confidence) / total
    
    def _generate_error_response(self, context: GenerationContext, error_message: str) -> Dict[str, Any]:
        """Generate error response"""
        template = self.template_manager.get_template(
            ResponseType.ERROR, 
            context.user_language,
            {"error_message": error_message}
        )
        
        response_text = self._fill_template(template, {"error_message": error_message})
        
        return {
            "text": response_text,
            "language": template.language,
            "response_type": ResponseType.ERROR.value,
            "template_id": template.template_id,
            "confidence": 0.5,
            "multimodal_elements": {"has_visual": False, "elements": []},
            "tone": template.tone,
            "metadata": {
                "is_error": True,
                "error_message": error_message
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get response generation statistics"""
        return self.stats.copy()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sample_context() -> GenerationContext:
    """Create sample generation context for testing"""
    return GenerationContext(
        intent="search",
        tool_results={
            "search": {
                "success": True,
                "result": {
                    "results": [
                        {
                            "title": "Wireless Headphones",
                            "price": 99.99,
                            "rating": 4.5,
                            "image_url": "https://example.com/headphones.jpg"
                        }
                    ],
                    "category": "electronics"
                }
            }
        },
        user_language="en",
        conversation_history=[],
        user_profile={"name": "John", "preferences": {"style": "modern"}},
        dialogue_state={"last_query": "wireless headphones"},
        multimodal_context={"has_images": False},
        confidence_score=0.9
    )
