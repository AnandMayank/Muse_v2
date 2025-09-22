#!/usr/bin/env python3
"""
MUSE v3 Data Generation Pipeline
===============================

Comprehensive data generation following best practices from:
- Toolformer: Self-supervised tool-call insertion and validation
- DiaTool: Paired trajectory format for DPO
- VisTA/ToRL: Tool-utility based reward construction
- AgentBench: Multi-task evaluation framework
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
import json
import random
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)

# =============================================================================
# 1. TOOLFORMER-STYLE SELF-SUPERVISION
# =============================================================================

@dataclass
class ToolCall:
    """Structured tool call representation"""
    tool_name: str
    arguments: Dict[str, Any]
    position: int  # Position in text where call should be inserted
    expected_output: Optional[str] = None
    confidence: float = 1.0
    cost: float = 1.0

@dataclass
class AugmentedContext:
    """Context with tool calls inserted"""
    original_text: str
    augmented_text: str
    tool_calls: List[ToolCall]
    quality_score: float
    metadata: Dict[str, Any]

class ToolformerDataGenerator:
    """
    Generate tool-augmented training data using Toolformer methodology
    
    Key innovations:
    1. Self-supervised tool call insertion
    2. Tool utility validation
    3. Naturalistic text generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        
        # Load language model for augmentation
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("lm_model", "microsoft/DialoGPT-medium")
        )
        self.language_model = AutoModel.from_pretrained(
            config.get("lm_model", "microsoft/DialoGPT-medium")
        )
        
        # Tool registry
        self.available_tools = self._setup_tools()
        
        # Toolformer-style templates
        self.tool_templates = {
            "search": "[SEARCH(query='{query}', category='{category}')]",
            "recommend": "[RECOMMEND(user_profile='{profile}', context='{context}')]",
            "compare": "[COMPARE(items=[{items}], aspects=[{aspects}])]",
            "filter": "[FILTER(items=[{items}], constraints={constraints})]",
            "translate": "[TRANSLATE(text='{text}', target_lang='{lang}')]",
            "visual_search": "[VISUAL_SEARCH(image_desc='{desc}', similarity={threshold})]"
        }
        
        logger.info("🔧 Toolformer-style data generator initialized")
    
    def _setup_tools(self) -> Dict[str, Dict]:
        """Setup available tools with cost and utility estimates"""
        return {
            "search": {
                "cost": 1.0,
                "avg_utility": 0.8,
                "success_rate": 0.9,
                "applicable_contexts": ["product_query", "information_need"]
            },
            "recommend": {
                "cost": 2.0,
                "avg_utility": 0.9,
                "success_rate": 0.85,
                "applicable_contexts": ["preference_expression", "browsing"]
            },
            "compare": {
                "cost": 1.5,
                "avg_utility": 0.85,
                "success_rate": 0.8,
                "applicable_contexts": ["multiple_items", "decision_making"]
            },
            "filter": {
                "cost": 0.5,
                "avg_utility": 0.7,
                "success_rate": 0.95,
                "applicable_contexts": ["refinement", "constraint_specification"]
            },
            "translate": {
                "cost": 0.3,
                "avg_utility": 0.95,
                "success_rate": 0.98,
                "applicable_contexts": ["multilingual", "language_switch"]
            },
            "visual_search": {
                "cost": 3.0,
                "avg_utility": 0.75,
                "success_rate": 0.7,
                "applicable_contexts": ["image_description", "visual_similarity"]
            }
        }
    
    def identify_tool_insertion_points(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify optimal points for tool call insertion using Toolformer approach
        
        Args:
            text: Input conversation text
            
        Returns:
            List of insertion points with tool recommendations
        """
        insertion_points = []
        
        # Pattern-based detection (simplified for demo)
        patterns = {
            "search": [
                r"(?i)(?:find|search|looking for|want to buy|need)",
                r"(?i)(?:show me|get me|where can i find)"
            ],
            "recommend": [
                r"(?i)(?:recommend|suggest|what should i)",
                r"(?i)(?:best|good|popular|trending)"
            ],
            "compare": [
                r"(?i)(?:compare|versus|vs|difference between)",
                r"(?i)(?:which is better|pros and cons)"
            ],
            "filter": [
                r"(?i)(?:under|within|less than|more than|\$[\d,]+)",
                r"(?i)(?:only|just|specifically|exactly)"
            ],
            "translate": [
                r"(?i)(?:in hindi|in english|translate)",
                r"[\u0900-\u097F]+"  # Devanagari script
            ],
            "visual_search": [
                r"(?i)(?:looks like|similar to|image|picture)",
                r"(?i)(?:visual|appearance|design)"
            ]
        }
        
        sentences = text.split('.')
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            for tool_name, tool_patterns in patterns.items():
                for pattern in tool_patterns:
                    if re.search(pattern, sentence):
                        # Calculate utility score for this insertion
                        utility = self._calculate_insertion_utility(
                            sentence, tool_name, i, len(sentences)
                        )
                        
                        if utility > self.config.get("min_utility_threshold", 0.5):
                            insertion_points.append({
                                "position": len('.'.join(sentences[:i+1])),
                                "sentence": sentence,
                                "tool_name": tool_name,
                                "utility": utility,
                                "context_window": sentences[max(0, i-1):i+2]
                            })
        
        # Sort by utility and filter top candidates
        insertion_points.sort(key=lambda x: x["utility"], reverse=True)
        return insertion_points[:self.config.get("max_insertions_per_text", 3)]
    
    def _calculate_insertion_utility(self, sentence: str, tool_name: str, 
                                   position: int, total_sentences: int) -> float:
        """Calculate utility of inserting a tool call at this position"""
        base_utility = self.available_tools[tool_name]["avg_utility"]
        
        # Position penalty (avoid too early or too late)
        position_factor = 1.0 - abs(0.3 - (position / total_sentences)) * 0.5
        
        # Context relevance (simplified keyword matching)
        context_keywords = self.available_tools[tool_name]["applicable_contexts"]
        relevance_score = sum(1 for keyword in context_keywords 
                            if keyword.lower() in sentence.lower()) / len(context_keywords)
        
        # Length penalty (avoid very short sentences)
        length_factor = min(1.0, len(sentence.split()) / 10.0)
        
        return base_utility * position_factor * (0.5 + relevance_score) * length_factor
    
    def generate_tool_call(self, insertion_point: Dict[str, Any]) -> ToolCall:
        """Generate structured tool call for insertion point"""
        tool_name = insertion_point["tool_name"]
        sentence = insertion_point["sentence"]
        
        # Extract arguments from context (simplified)
        arguments = self._extract_arguments(sentence, tool_name)
        
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            position=insertion_point["position"],
            confidence=insertion_point["utility"],
            cost=self.available_tools[tool_name]["cost"]
        )
    
    def _extract_arguments(self, sentence: str, tool_name: str) -> Dict[str, Any]:
        """Extract tool arguments from sentence context"""
        args = {}
        
        if tool_name == "search":
            # Extract search query and category
            query_match = re.search(r'(?:find|search|looking for|want|need)\s+(.+?)(?:[.!?]|$)', sentence, re.I)
            args["query"] = query_match.group(1).strip() if query_match else sentence[:50]
            args["category"] = "general"  # Could be improved with NER
            args["max_results"] = 10
        
        elif tool_name == "recommend":
            args["num_items"] = 5
            args["user_profile"] = "general"
            args["context"] = sentence[:100]
        
        elif tool_name == "compare":
            # Extract items to compare
            items = re.findall(r'\b(?:iphone|samsung|nike|adidas|[\w\s]{2,20})\b', sentence, re.I)
            args["items"] = items[:3] if items else ["item1", "item2"]
            args["comparison_aspects"] = ["price", "features", "reviews"]
        
        elif tool_name == "filter":
            # Extract constraints
            price_match = re.search(r'\$?(\d+(?:,\d{3})*)', sentence)
            args["max_price"] = int(price_match.group(1).replace(',', '')) if price_match else 1000
            args["filters"] = {"price_range": f"under_{args['max_price']}"}
        
        elif tool_name == "translate":
            args["text"] = sentence
            args["source_lang"] = "auto"
            args["target_lang"] = "hi" if any(ord(c) > 127 for c in sentence) else "en"
        
        elif tool_name == "visual_search":
            args["similarity_threshold"] = 0.8
            args["image_description"] = sentence[:100]
        
        return args
    
    def create_augmented_context(self, original_text: str) -> AugmentedContext:
        """
        Create Toolformer-style augmented context with tool calls
        
        Args:
            original_text: Original conversation context
            
        Returns:
            Augmented context with inserted tool calls
        """
        insertion_points = self.identify_tool_insertion_points(original_text)
        
        if not insertion_points:
            return AugmentedContext(
                original_text=original_text,
                augmented_text=original_text,
                tool_calls=[],
                quality_score=0.0,
                metadata={"reason": "no_suitable_insertions"}
            )
        
        # Generate tool calls
        tool_calls = [self.generate_tool_call(point) for point in insertion_points]
        
        # Insert tool calls into text
        augmented_text = self._insert_tool_calls(original_text, tool_calls)
        
        # Calculate quality score
        quality_score = self._evaluate_augmentation_quality(
            original_text, augmented_text, tool_calls
        )
        
        return AugmentedContext(
            original_text=original_text,
            augmented_text=augmented_text,
            tool_calls=tool_calls,
            quality_score=quality_score,
            metadata={
                "num_insertions": len(tool_calls),
                "avg_utility": np.mean([call.confidence for call in tool_calls]),
                "total_cost": sum(call.cost for call in tool_calls)
            }
        )
    
    def _insert_tool_calls(self, text: str, tool_calls: List[ToolCall]) -> str:
        """Insert tool calls into text at appropriate positions"""
        # Sort by position in reverse order to maintain positions
        sorted_calls = sorted(tool_calls, key=lambda x: x.position, reverse=True)
        
        augmented = text
        for call in sorted_calls:
            # Generate tool call string
            template = self.tool_templates[call.tool_name]
            
            # Format arguments into template
            try:
                if call.tool_name == "search":
                    call_str = template.format(
                        query=call.arguments.get("query", ""),
                        category=call.arguments.get("category", "general")
                    )
                elif call.tool_name == "recommend":
                    call_str = template.format(
                        profile=call.arguments.get("user_profile", "general"),
                        context=call.arguments.get("context", "")[:50]
                    )
                else:
                    # Generic formatting
                    call_str = f"[{call.tool_name.upper()}({str(call.arguments)})]"
                
                # Insert at position
                augmented = augmented[:call.position] + " " + call_str + " " + augmented[call.position:]
                
            except Exception as e:
                logger.warning(f"Failed to format tool call: {e}")
                continue
        
        return augmented.strip()
    
    def _evaluate_augmentation_quality(self, original: str, augmented: str, 
                                     tool_calls: List[ToolCall]) -> float:
        """Evaluate quality of tool call augmentation"""
        if not tool_calls:
            return 0.0
        
        # Factors for quality scoring
        factors = {
            "utility": np.mean([call.confidence for call in tool_calls]),
            "naturalness": min(1.0, len(original) / len(augmented)),  # Avoid over-augmentation
            "diversity": len(set(call.tool_name for call in tool_calls)) / len(tool_calls),
            "cost_efficiency": 1.0 / (1.0 + sum(call.cost for call in tool_calls) / len(tool_calls))
        }
        
        # Weighted combination
        weights = {"utility": 0.4, "naturalness": 0.3, "diversity": 0.2, "cost_efficiency": 0.1}
        quality = sum(weights[factor] * score for factor, score in factors.items())
        
        return min(1.0, quality)
    
    def batch_generate_toolformer_data(self, contexts: List[str], 
                                     output_path: str) -> Dict[str, Any]:
        """
        Generate Toolformer-style training data from MUSE contexts
        
        Args:
            contexts: List of conversation contexts
            output_path: Where to save generated data
            
        Returns:
            Generation statistics
        """
        logger.info(f"🔄 Generating Toolformer-style data from {len(contexts)} contexts...")
        
        augmented_data = []
        stats = {"total": 0, "augmented": 0, "tool_calls": 0, "avg_quality": 0.0}
        
        for context in contexts:
            try:
                augmented = self.create_augmented_context(context)
                augmented_data.append(asdict(augmented))
                
                stats["total"] += 1
                if augmented.tool_calls:
                    stats["augmented"] += 1
                    stats["tool_calls"] += len(augmented.tool_calls)
                    stats["avg_quality"] += augmented.quality_score
                    
            except Exception as e:
                logger.error(f"Failed to augment context: {e}")
                continue
        
        # Calculate final statistics
        if stats["augmented"] > 0:
            stats["avg_quality"] /= stats["augmented"]
            stats["avg_calls_per_context"] = stats["tool_calls"] / stats["augmented"]
        
        # Save data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated {stats['augmented']}/{stats['total']} augmented contexts")
        logger.info(f"   📊 Avg quality: {stats['avg_quality']:.3f}")
        logger.info(f"   🔧 Total tool calls: {stats['tool_calls']}")
        
        return stats

# =============================================================================
# 2. DPO PAIR CONSTRUCTION (DiaTool Style)
# =============================================================================

@dataclass
class DPOPair:
    """DPO training pair following DiaTool format"""
    context: str
    chosen_trajectory: List[Dict[str, Any]]
    rejected_trajectory: List[Dict[str, Any]]
    preference_score: float  # How much better is chosen vs rejected
    metadata: Dict[str, Any]

class DPOPairGenerator:
    """
    Generate DPO training pairs following DiaTool methodology
    
    Key components:
    1. Trajectory pair construction
    2. Preference annotation
    3. Quality validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        
        # Load models for trajectory generation
        self.paraphraser = T5ForConditionalGeneration.from_pretrained(
            config.get("paraphraser_model", "t5-small")
        )
        self.paraphraser_tokenizer = AutoTokenizer.from_pretrained(
            config.get("paraphraser_model", "t5-small")
        )
        
        logger.info("🎯 DPO pair generator initialized")
    
    def create_rejected_variants(self, chosen_trajectory: List[Dict[str, Any]], 
                               num_variants: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Create rejected trajectory variants using multiple strategies
        
        Rejection types:
        - Type 1: Wrong tool selection
        - Type 2: Correct tool, wrong arguments  
        - Type 3: Right tool/args, poor execution order
        """
        rejected_variants = []
        
        for i in range(num_variants):
            variant_type = i % 3  # Cycle through rejection types
            
            if variant_type == 0:  # Wrong tool selection
                rejected = self._create_wrong_tool_variant(chosen_trajectory)
            elif variant_type == 1:  # Wrong arguments
                rejected = self._create_wrong_args_variant(chosen_trajectory)
            else:  # Poor execution order
                rejected = self._create_wrong_order_variant(chosen_trajectory)
            
            if rejected:
                rejected_variants.append(rejected)
        
        return rejected_variants
    
    def _create_wrong_tool_variant(self, chosen: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create variant with wrong tool selection"""
        rejected = deepcopy(chosen)
        
        # Find tool calls and replace with inappropriate alternatives
        tool_alternatives = {
            "search": ["recommend", "compare"],
            "recommend": ["search", "filter"], 
            "compare": ["search", "translate"],
            "filter": ["search", "recommend"],
            "translate": ["search", "visual_search"],
            "visual_search": ["search", "compare"]
        }
        
        for step in rejected:
            if "tool_call" in step:
                original_tool = step["tool_call"]["tool_name"]
                if original_tool in tool_alternatives:
                    alternatives = tool_alternatives[original_tool]
                    step["tool_call"]["tool_name"] = random.choice(alternatives)
                    # Keep same arguments (which will be inappropriate for new tool)
        
        return rejected
    
    def _create_wrong_args_variant(self, chosen: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create variant with wrong tool arguments"""
        rejected = deepcopy(chosen)
        
        for step in rejected:
            if "tool_call" in step:
                args = step["tool_call"]["arguments"]
                
                # Introduce argument errors
                if "query" in args:
                    args["query"] = args["query"][::-1]  # Reverse query
                if "max_results" in args:
                    args["max_results"] = 1  # Too restrictive
                if "similarity_threshold" in args:
                    args["similarity_threshold"] = 0.1  # Too low threshold
                if "num_items" in args:
                    args["num_items"] = 100  # Too many items
        
        return rejected
    
    def _create_wrong_order_variant(self, chosen: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create variant with poor execution order"""
        rejected = deepcopy(chosen)
        
        # Shuffle middle steps (keep first and last in place)
        if len(rejected) > 3:
            middle_steps = rejected[1:-1]
            random.shuffle(middle_steps)
            rejected = [rejected[0]] + middle_steps + [rejected[-1]]
        
        return rejected
    
    def calculate_preference_score(self, chosen: List[Dict[str, Any]], 
                                 rejected: List[Dict[str, Any]], 
                                 context: str) -> float:
        """
        Calculate preference score between chosen and rejected trajectories
        
        Based on:
        - Tool appropriateness
        - Argument quality  
        - Execution efficiency
        - Expected utility
        """
        chosen_score = self._score_trajectory(chosen, context)
        rejected_score = self._score_trajectory(rejected, context)
        
        # Preference margin (how much better chosen is)
        preference = (chosen_score - rejected_score) / (chosen_score + rejected_score + 1e-8)
        return max(0.0, preference)  # Ensure non-negative preference
    
    def _score_trajectory(self, trajectory: List[Dict[str, Any]], context: str) -> float:
        """Score a single trajectory for quality"""
        if not trajectory:
            return 0.0
        
        scores = {
            "tool_appropriateness": 0.0,
            "argument_quality": 0.0,
            "execution_efficiency": 0.0,
            "expected_utility": 0.0
        }
        
        for step in trajectory:
            if "tool_call" in step:
                tool_name = step["tool_call"]["tool_name"]
                
                # Tool appropriateness (context matching)
                if self._tool_matches_context(tool_name, context):
                    scores["tool_appropriateness"] += 1.0
                
                # Argument quality (basic validation)
                args = step["tool_call"]["arguments"]
                if self._validate_arguments(tool_name, args):
                    scores["argument_quality"] += 1.0
                
                # Expected utility (from tool registry)
                scores["expected_utility"] += self._get_tool_utility(tool_name)
        
        # Execution efficiency (fewer steps = better)
        scores["execution_efficiency"] = max(0.0, 1.0 - len(trajectory) / 10.0)
        
        # Weighted combination
        weights = {
            "tool_appropriateness": 0.3,
            "argument_quality": 0.3, 
            "execution_efficiency": 0.2,
            "expected_utility": 0.2
        }
        
        total_score = sum(weights[key] * score for key, score in scores.items())
        return total_score
    
    def _tool_matches_context(self, tool_name: str, context: str) -> bool:
        """Check if tool is appropriate for context"""
        context_lower = context.lower()
        
        tool_patterns = {
            "search": ["find", "search", "looking for", "want", "need"],
            "recommend": ["recommend", "suggest", "best", "good", "popular"],
            "compare": ["compare", "vs", "versus", "difference", "better"],
            "filter": ["under", "within", "less than", "only", "specific"],
            "translate": ["hindi", "english", "translate"] + [chr(i) for i in range(0x0900, 0x097F)],
            "visual_search": ["looks like", "similar", "image", "picture", "visual"]
        }
        
        if tool_name in tool_patterns:
            return any(pattern in context_lower for pattern in tool_patterns[tool_name])
        return False
    
    def _validate_arguments(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Basic argument validation"""
        if tool_name == "search":
            return "query" in args and isinstance(args["query"], str) and len(args["query"]) > 0
        elif tool_name == "recommend":
            return "num_items" in args and isinstance(args["num_items"], int) and args["num_items"] > 0
        elif tool_name == "filter":
            return "filters" in args or "max_price" in args
        # Add more validations as needed
        return True
    
    def _get_tool_utility(self, tool_name: str) -> float:
        """Get expected utility for tool (from ToolformerDataGenerator)"""
        utilities = {
            "search": 0.8, "recommend": 0.9, "compare": 0.85,
            "filter": 0.7, "translate": 0.95, "visual_search": 0.75
        }
        return utilities.get(tool_name, 0.5)
    
    def generate_dpo_pair(self, context: str, 
                         chosen_trajectory: List[Dict[str, Any]]) -> List[DPOPair]:
        """Generate DPO training pairs from context and chosen trajectory"""
        rejected_variants = self.create_rejected_variants(chosen_trajectory)
        
        dpo_pairs = []
        for rejected in rejected_variants:
            preference_score = self.calculate_preference_score(
                chosen_trajectory, rejected, context
            )
            
            # Only keep pairs with sufficient preference margin
            if preference_score > self.config.get("min_preference_score", 0.1):
                pair = DPOPair(
                    context=context,
                    chosen_trajectory=chosen_trajectory,
                    rejected_trajectory=rejected,
                    preference_score=preference_score,
                    metadata={
                        "chosen_score": self._score_trajectory(chosen_trajectory, context),
                        "rejected_score": self._score_trajectory(rejected, context),
                        "rejection_type": self._identify_rejection_type(chosen_trajectory, rejected)
                    }
                )
                dpo_pairs.append(pair)
        
        return dpo_pairs
    
    def _identify_rejection_type(self, chosen: List[Dict[str, Any]], 
                               rejected: List[Dict[str, Any]]) -> str:
        """Identify what type of rejection this represents"""
        # Compare tool names
        chosen_tools = [step.get("tool_call", {}).get("tool_name") for step in chosen if "tool_call" in step]
        rejected_tools = [step.get("tool_call", {}).get("tool_name") for step in rejected if "tool_call" in step]
        
        if chosen_tools != rejected_tools:
            return "wrong_tool"
        
        # Compare arguments
        for c_step, r_step in zip(chosen, rejected):
            if "tool_call" in c_step and "tool_call" in r_step:
                if c_step["tool_call"]["arguments"] != r_step["tool_call"]["arguments"]:
                    return "wrong_arguments"
        
        return "wrong_order"

# =============================================================================
# 3. VISTA/TORL REWARD CONSTRUCTION  
# =============================================================================

@dataclass
class ToolReward:
    """Tool-specific reward components"""
    utility_delta: float  # Change in task utility
    cost_penalty: float   # Cost penalty for tool usage
    success_bonus: float  # Bonus for successful execution
    efficiency_bonus: float  # Bonus for efficient execution
    total_reward: float

class VisTA_RewardCalculator:
    """
    Calculate VisTA/ToRL style rewards for tool selection
    
    Components:
    1. Tool utility delta measurement
    2. Cost-aware penalty system
    3. Success rate bonuses
    4. Efficiency rewards
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Reward hyperparameters (following VisTA)
        self.utility_weight = config.get("utility_weight", 1.0)
        self.cost_weight = config.get("cost_weight", 0.2)
        self.success_weight = config.get("success_weight", 0.5)
        self.efficiency_weight = config.get("efficiency_weight", 0.3)
        
        logger.info("🎁 VisTA-style reward calculator initialized")
    
    def calculate_tool_reward(self, 
                            tool_name: str,
                            arguments: Dict[str, Any],
                            execution_result: Dict[str, Any],
                            context: Dict[str, Any]) -> ToolReward:
        """
        Calculate comprehensive reward for tool usage
        
        Args:
            tool_name: Name of the tool used
            arguments: Arguments passed to tool
            execution_result: Result of tool execution
            context: Task context and state
            
        Returns:
            Structured reward breakdown
        """
        # 1. Utility delta calculation
        utility_delta = self._calculate_utility_delta(
            tool_name, execution_result, context
        )
        
        # 2. Cost penalty
        cost_penalty = self._calculate_cost_penalty(tool_name, arguments)
        
        # 3. Success bonus
        success_bonus = self._calculate_success_bonus(execution_result)
        
        # 4. Efficiency bonus  
        efficiency_bonus = self._calculate_efficiency_bonus(
            tool_name, arguments, execution_result, context
        )
        
        # Total reward combination
        total_reward = (
            self.utility_weight * utility_delta -
            self.cost_weight * cost_penalty +
            self.success_weight * success_bonus +
            self.efficiency_weight * efficiency_bonus
        )
        
        return ToolReward(
            utility_delta=utility_delta,
            cost_penalty=cost_penalty,
            success_bonus=success_bonus,
            efficiency_bonus=efficiency_bonus,
            total_reward=total_reward
        )
    
    def _calculate_utility_delta(self, tool_name: str, 
                               execution_result: Dict[str, Any],
                               context: Dict[str, Any]) -> float:
        """Calculate change in task utility from tool usage"""
        # Task utility before tool usage
        pre_utility = context.get("task_progress", 0.0)
        
        # Estimated utility after tool usage
        post_utility = pre_utility
        
        if execution_result.get("success", False):
            # Tool-specific utility gains
            utility_gains = {
                "search": 0.3,      # Found relevant items
                "recommend": 0.4,   # Provided personalized suggestions  
                "compare": 0.25,    # Helped decision making
                "filter": 0.15,     # Narrowed down options
                "translate": 0.2,   # Enabled communication
                "visual_search": 0.35  # Found visually similar items
            }
            
            base_gain = utility_gains.get(tool_name, 0.1)
            
            # Adjust based on result quality
            quality_multiplier = execution_result.get("quality_score", 0.5)
            post_utility += base_gain * quality_multiplier
        
        return max(0.0, post_utility - pre_utility)
    
    def _calculate_cost_penalty(self, tool_name: str, 
                              arguments: Dict[str, Any]) -> float:
        """Calculate cost penalty for tool usage"""
        # Base costs (from Toolformer setup)
        base_costs = {
            "search": 1.0,
            "recommend": 2.0,
            "compare": 1.5,
            "filter": 0.5,
            "translate": 0.3,
            "visual_search": 3.0
        }
        
        base_cost = base_costs.get(tool_name, 1.0)
        
        # Argument-based cost modifiers
        cost_multiplier = 1.0
        
        if tool_name == "search":
            max_results = arguments.get("max_results", 10)
            cost_multiplier = 1.0 + (max_results - 10) * 0.05
        elif tool_name == "recommend":
            num_items = arguments.get("num_items", 5)
            cost_multiplier = 1.0 + (num_items - 5) * 0.1
        elif tool_name == "visual_search":
            # High-res image processing costs more
            cost_multiplier = 1.2
        
        return base_cost * cost_multiplier
    
    def _calculate_success_bonus(self, execution_result: Dict[str, Any]) -> float:
        """Calculate bonus for successful tool execution"""
        if not execution_result.get("success", False):
            return 0.0
        
        # Base success bonus
        base_bonus = 0.5
        
        # Quality-based multiplier
        quality_score = execution_result.get("quality_score", 0.5)
        quality_multiplier = 0.5 + quality_score  # 0.5 to 1.5 range
        
        return base_bonus * quality_multiplier
    
    def _calculate_efficiency_bonus(self, tool_name: str,
                                  arguments: Dict[str, Any],
                                  execution_result: Dict[str, Any],
                                  context: Dict[str, Any]) -> float:
        """Calculate efficiency bonus for optimal tool usage"""
        efficiency_score = 0.0
        
        # Time efficiency (faster execution = bonus)
        execution_time = execution_result.get("execution_time", 1.0)
        time_bonus = max(0.0, (2.0 - execution_time) * 0.2)  # Bonus for < 2s execution
        
        # Result relevance efficiency  
        relevance = execution_result.get("relevance_score", 0.5)
        relevance_bonus = relevance * 0.3
        
        # Context appropriateness
        appropriateness = self._measure_context_appropriateness(tool_name, context)
        context_bonus = appropriateness * 0.2
        
        efficiency_score = time_bonus + relevance_bonus + context_bonus
        
        return efficiency_score
    
    def _measure_context_appropriateness(self, tool_name: str, 
                                       context: Dict[str, Any]) -> float:
        """Measure how appropriate tool choice is for current context"""
        # Simple rule-based appropriateness
        context_text = context.get("user_input", "").lower()
        current_intent = context.get("intent", "")
        
        appropriateness_map = {
            ("search", "search"): 1.0,
            ("recommend", "recommend"): 1.0,
            ("compare", "compare"): 1.0,
            ("filter", "filter"): 1.0,
            ("translate", "translate"): 1.0,
            ("visual_search", "visual_search"): 1.0,
        }
        
        return appropriateness_map.get((tool_name, current_intent), 0.5)

# =============================================================================
# 4. MAIN DATA GENERATION ORCHESTRATOR
# =============================================================================

class DataGenerationPipeline:
    """
    Main orchestrator for comprehensive data generation
    
    Combines:
    1. Toolformer-style self-supervision
    2. DPO pair construction  
    3. VisTA reward calculation
    4. Quality validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.toolformer = ToolformerDataGenerator(config)
        self.dpo_generator = DPOPairGenerator(config)
        self.reward_calculator = VisTA_RewardCalculator(config)
        
        # Output paths
        self.output_dir = Path(config.get("output_dir", "./generated_data"))
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Complete data generation pipeline initialized")
    
    def run_full_pipeline(self, muse_contexts: List[str]) -> Dict[str, Any]:
        """
        Run complete data generation pipeline
        
        Args:
            muse_contexts: Original MUSE conversation contexts
            
        Returns:
            Pipeline execution statistics
        """
        logger.info(f"🏃‍♂️ Running full data generation pipeline on {len(muse_contexts)} contexts")
        
        results = {}
        
        # Phase 1: Toolformer-style augmentation
        logger.info("📝 Phase 1: Toolformer-style data augmentation")
        toolformer_stats = self.toolformer.batch_generate_toolformer_data(
            muse_contexts,
            str(self.output_dir / "toolformer_augmented.json")
        )
        results["toolformer"] = toolformer_stats
        
        # Phase 2: DPO pair generation (using augmented data)
        logger.info("🎯 Phase 2: DPO pair construction")
        dpo_stats = self._generate_dpo_dataset()
        results["dpo"] = dpo_stats
        
        # Phase 3: Reward calculation for RL
        logger.info("🎁 Phase 3: VisTA-style reward calculation")  
        reward_stats = self._generate_reward_dataset()
        results["rewards"] = reward_stats
        
        # Phase 4: Quality validation and filtering
        logger.info("✅ Phase 4: Quality validation")
        validation_stats = self._validate_generated_data()
        results["validation"] = validation_stats
        
        # Save comprehensive results
        with open(self.output_dir / "generation_report.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("🎉 Data generation pipeline completed successfully!")
        logger.info(f"📊 Generated data saved to: {self.output_dir}")
        
        return results
    
    def _generate_dpo_dataset(self) -> Dict[str, Any]:
        """Generate DPO dataset from augmented contexts"""
        # Load augmented data
        with open(self.output_dir / "toolformer_augmented.json", 'r') as f:
            augmented_data = json.load(f)
        
        dpo_pairs = []
        for item in augmented_data:
            if not item["tool_calls"]:
                continue
            
            # Convert tool calls to trajectory format
            chosen_trajectory = []
            for tool_call in item["tool_calls"]:
                chosen_trajectory.append({
                    "tool_call": {
                        "tool_name": tool_call["tool_name"],
                        "arguments": tool_call["arguments"]
                    },
                    "position": tool_call["position"]
                })
            
            # Generate DPO pairs
            pairs = self.dpo_generator.generate_dpo_pair(
                item["original_text"], chosen_trajectory
            )
            dpo_pairs.extend(pairs)
        
        # Save DPO dataset
        dpo_data = [asdict(pair) for pair in dpo_pairs]
        with open(self.output_dir / "dpo_pairs.json", 'w') as f:
            json.dump(dpo_data, f, indent=2)
        
        return {
            "total_pairs": len(dpo_pairs),
            "avg_preference_score": np.mean([p.preference_score for p in dpo_pairs]),
            "rejection_types": {
                "wrong_tool": len([p for p in dpo_pairs if p.metadata["rejection_type"] == "wrong_tool"]),
                "wrong_arguments": len([p for p in dpo_pairs if p.metadata["rejection_type"] == "wrong_arguments"]),
                "wrong_order": len([p for p in dpo_pairs if p.metadata["rejection_type"] == "wrong_order"])
            }
        }
    
    def _generate_reward_dataset(self) -> Dict[str, Any]:
        """Generate VisTA-style reward dataset"""
        # Load DPO pairs for reward calculation
        with open(self.output_dir / "dpo_pairs.json", 'r') as f:
            dpo_data = json.load(f)
        
        reward_data = []
        for pair_data in dpo_data:
            context = {"user_input": pair_data["context"], "task_progress": 0.0}
            
            # Calculate rewards for chosen trajectory
            for step in pair_data["chosen_trajectory"]:
                if "tool_call" in step:
                    # Mock execution result
                    execution_result = {
                        "success": True,
                        "quality_score": 0.8,
                        "execution_time": 1.0,
                        "relevance_score": 0.7
                    }
                    
                    reward = self.reward_calculator.calculate_tool_reward(
                        step["tool_call"]["tool_name"],
                        step["tool_call"]["arguments"],
                        execution_result,
                        context
                    )
                    
                    reward_data.append({
                        "context": pair_data["context"],
                        "tool_name": step["tool_call"]["tool_name"],
                        "arguments": step["tool_call"]["arguments"],
                        "reward": asdict(reward)
                    })
        
        # Save reward dataset
        with open(self.output_dir / "reward_dataset.json", 'w') as f:
            json.dump(reward_data, f, indent=2)
        
        return {
            "total_rewards": len(reward_data),
            "avg_total_reward": np.mean([r["reward"]["total_reward"] for r in reward_data]),
            "avg_utility_delta": np.mean([r["reward"]["utility_delta"] for r in reward_data])
        }
    
    def _validate_generated_data(self) -> Dict[str, Any]:
        """Validate quality of generated data"""
        validation_results = {
            "toolformer_quality": self._validate_toolformer_data(),
            "dpo_quality": self._validate_dpo_data(), 
            "reward_quality": self._validate_reward_data()
        }
        
        return validation_results
    
    def _validate_toolformer_data(self) -> Dict[str, Any]:
        """Validate Toolformer-generated data"""
        with open(self.output_dir / "toolformer_augmented.json", 'r') as f:
            data = json.load(f)
        
        quality_scores = [item["quality_score"] for item in data if item["tool_calls"]]
        
        return {
            "avg_quality_score": np.mean(quality_scores) if quality_scores else 0.0,
            "high_quality_ratio": len([q for q in quality_scores if q > 0.7]) / max(1, len(quality_scores)),
            "total_augmented": len([item for item in data if item["tool_calls"]])
        }
    
    def _validate_dpo_data(self) -> Dict[str, Any]:
        """Validate DPO pair quality"""
        with open(self.output_dir / "dpo_pairs.json", 'r') as f:
            data = json.load(f)
        
        preference_scores = [item["preference_score"] for item in data]
        
        return {
            "avg_preference_score": np.mean(preference_scores) if preference_scores else 0.0,
            "strong_preference_ratio": len([p for p in preference_scores if p > 0.5]) / max(1, len(preference_scores)),
            "total_pairs": len(data)
        }
    
    def _validate_reward_data(self) -> Dict[str, Any]:
        """Validate reward calculation quality"""
        with open(self.output_dir / "reward_dataset.json", 'r') as f:
            data = json.load(f)
        
        total_rewards = [item["reward"]["total_reward"] for item in data]
        
        return {
            "avg_total_reward": np.mean(total_rewards) if total_rewards else 0.0,
            "positive_reward_ratio": len([r for r in total_rewards if r > 0]) / max(1, len(total_rewards)),
            "total_reward_samples": len(data)
        }

# =============================================================================
# TESTING & DEMONSTRATION
# =============================================================================

def test_data_generation_pipeline():
    """Test the complete data generation pipeline"""
    print("🧪 Testing Data Generation Pipeline")
    print("=" * 50)
    
    # Configuration
    config = {
        "device": "cpu",
        "min_utility_threshold": 0.3,
        "max_insertions_per_text": 2,
        "min_preference_score": 0.1,
        "output_dir": "/media/adityapachauri/second_drive/Muse/muse_v3_advanced/generated_data"
    }
    
    # Sample MUSE contexts
    sample_contexts = [
        "I'm looking for running shoes under $100. I prefer Nike or Adidas brands.",
        "Can you recommend a good smartphone for photography? I mostly take pictures of food and travel.",
        "मुझे एक अच्छी किताब चाहिए। कुछ रोमांटिक या mystery genre में।",
        "I want to compare iPhone 15 vs Samsung Galaxy S24. Which has better camera?",
        "Show me dresses similar to this image that are suitable for office wear."
    ]
    
    # Initialize pipeline
    pipeline = DataGenerationPipeline(config)
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(sample_contexts)
    
    print("\n📊 Pipeline Results:")
    print(f"   🔧 Toolformer augmentations: {results['toolformer']['augmented']}")
    print(f"   🎯 DPO pairs generated: {results['dpo']['total_pairs']}")
    print(f"   🎁 Reward samples: {results['rewards']['total_rewards']}")
    print(f"   ✅ Validation passed: {results['validation']}")
    
    return True

if __name__ == "__main__":
    test_data_generation_pipeline()
