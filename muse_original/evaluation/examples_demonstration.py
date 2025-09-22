#!/usr/bin/env python3
"""
Example: Enhanced MUSe v2 Key Improvements Demonstration
This script showcases the major enhancements over the original MUSe system
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Any


class MuseV2Demonstration:
    """Demonstrates key improvements in MUSe v2"""
    
    def __init__(self):
        self.examples = {
            "data_quality": [],
            "richer_context": [],
            "tool_ready": [],
            "training_pipeline": []
        }
    
    def demonstrate_data_quality_improvements(self):
        """Show data quality and realism improvements"""
        print("🧹 Data Quality & Realism Improvements")
        print("="*50)
        
        # 1. Metadata Normalization Example
        print("\n1. Metadata Normalization:")
        original_item = {
            "title": "Women's T-Shirt",
            "categories": ["t-shirt", "tshirt", "shirt", "tops"],  # Duplicates
            "price": "$29.99 USD",  # String format
            "sizes": ["extra small", "S", "medium", "L", "xlarge"],  # Inconsistent
            "color": "Blue"
        }
        
        normalized_item = {
            "title": "Women's T-Shirt",
            "categories": ["tops"],  # Deduplicated and normalized
            "price": 29.99,  # Numeric format
            "sizes": ["XS", "S", "M", "L", "XL"],  # Consistent format
            "color": "blue",  # Lowercase normalized
            "item_id": "item_12345",  # Ensured required fields
            "availability": True
        }
        
        print(f"Original: {json.dumps(original_item, indent=2)}")
        print(f"Normalized: {json.dumps(normalized_item, indent=2)}")
        
        # 2. Natural Dialogue Variations
        print("\n2. Natural Dialogue Variations:")
        basic_response = "I recommend this blue dress for you."
        
        enhanced_responses = [
            "Hmm, let me think... I believe this blue dress would be perfect for you!",
            "Actually, let me suggest this beautiful blue dress - I think you'll love it.",
            "Here's something that caught my attention: this elegant blue dress.",
            "Perfect! I found exactly what you're looking for - this stunning blue dress."
        ]
        
        print(f"Basic: {basic_response}")
        print("Enhanced with variations:")
        for response in enhanced_responses:
            print(f"  • {response}")
        
        # 3. Failure Recovery Example
        print("\n3. Failure Recovery Scenarios:")
        failure_scenario = {
            "original_search": "blue dress under $100",
            "failure": {
                "type": "no_results_found",
                "error": "No items found matching your criteria",
                "suggestion": "Try broadening your search terms"
            },
            "recovery_turn": "I apologize - I couldn't find blue dresses under $100. Would you like me to show you blue dresses in a slightly higher price range, or perhaps other colors under $100?",
            "alternative_approach": "Let me try searching for 'affordable blue clothing' instead"
        }
        
        print(json.dumps(failure_scenario, indent=2))
        
        self.examples["data_quality"] = {
            "metadata_normalization": {"original": original_item, "normalized": normalized_item},
            "dialogue_variations": enhanced_responses,
            "failure_recovery": failure_scenario
        }
    
    def demonstrate_richer_context(self):
        """Show richer context with personas and goals"""
        print("\n👤 Richer Context with Personas & Goals")
        print("="*50)
        
        # 1. Synthetic User Persona
        print("\n1. Enhanced User Persona:")
        persona_example = {
            "user_id": "user_42031",
            "demographics": {
                "age_range": "25-34",
                "occupation": "Marketing Manager",
                "lifestyle": "Urban Professional"
            },
            "preferences": {
                "style": ["minimalist", "casual"],
                "budget_range": "50-200",
                "brands": ["Uniqlo", "Zara", "Gap"],
                "colors": ["black", "white", "navy", "gray"],
                "sizes": {"top": "M", "bottom": "M", "shoes": "8"}
            },
            "personality": {
                "traits": ["Practical", "Quality-focused", "Detail-oriented"],
                "shopping_frequency": "Monthly"
            },
            "context": {
                "purchase_history": [
                    {
                        "item_id": "item_1234",
                        "category": "Tops",
                        "price": 45.00,
                        "satisfaction_rating": 4.5,
                        "context": "Work wardrobe update"
                    }
                ],
                "wardrobe_gaps": ["Professional blazer", "Comfortable shoes"],
                "upcoming_events": ["Job interview", "Conference"],
                "shopping_goals": ["Build professional wardrobe", "Find versatile pieces"]
            }
        }
        
        print(json.dumps(persona_example, indent=2))
        
        # 2. Contextual Dialogue Goals
        print("\n2. Specific Dialogue Goals:")
        dialogue_goals = [
            {
                "goal_type": "find_specific_item",
                "goal_text": "Find a professional blazer in navy under $150",
                "success_criteria": ["Exact color match", "Under budget limit", "Professional style"],
                "difficulty": "medium",
                "expected_turns": 5
            },
            {
                "goal_type": "special_occasion",
                "goal_text": "Find an outfit for job interview that matches minimalist style",
                "success_criteria": ["Appropriate for occasion", "Matches personal style", "Professional appearance"],
                "difficulty": "hard",
                "expected_turns": 7
            }
        ]
        
        for goal in dialogue_goals:
            print(json.dumps(goal, indent=2))
            print()
        
        # 3. Intent Labeling
        print("\n3. Comprehensive Intent Labeling:")
        intent_examples = [
            {"turn": "Hi, I'm looking for something professional", "intent": "search"},
            {"turn": "Can you compare these two blazers?", "intent": "compare"},
            {"turn": "I need something under $100", "intent": "filter"},
            {"turn": "That's perfect! I'll take it", "intent": "accept"},
            {"turn": "Not quite right, too formal", "intent": "reject"},
            {"turn": "That reminds me of my old job", "intent": "chit_chat"}
        ]
        
        for example in intent_examples:
            print(f"'{example['turn']}' → {example['intent']}")
        
        self.examples["richer_context"] = {
            "persona": persona_example,
            "dialogue_goals": dialogue_goals,
            "intent_labeling": intent_examples
        }
    
    def demonstrate_tool_ready_architecture(self):
        """Show tool-ready architecture for training"""
        print("\n🛠 Tool-Ready Architecture")
        print("="*50)
        
        # 1. Tool Call Integration
        print("\n1. Explicit Tool Calls:")
        tool_call_example = {
            "context": "User: Show me blue dresses under $100",
            "chosen_tool": {
                "tool": "search_items",
                "args": {
                    "query": "blue dress",
                    "filters": {"max_price": 100, "category": "dresses"},
                    "limit": 5
                },
                "success_rate": 0.95
            },
            "alternative_tools": [
                {
                    "tool": "filter_items",
                    "args": {"color": "blue", "price_range": [0, 100]},
                    "success_rate": 0.60
                },
                {
                    "tool": "visual_search",
                    "args": {"color_features": ["blue"], "style": "dress"},
                    "success_rate": 0.70
                }
            ],
            "response": "I found several beautiful blue dresses under $100. Here's one that matches your style perfectly!"
        }
        
        print(json.dumps(tool_call_example, indent=2))
        
        # 2. Available Tools
        print("\n2. Available Tool Set:")
        available_tools = {
            "search_items": "Search for items based on criteria",
            "fetch_details": "Get detailed information about specific items",
            "compare_items": "Compare multiple items side by side",
            "filter_items": "Filter existing results by criteria",
            "recommend_similar": "Find items similar to a given item",
            "check_availability": "Check if item is in stock",
            "get_reviews": "Retrieve user reviews for items",
            "visual_search": "Search using visual features and style"
        }
        
        for tool, description in available_tools.items():
            print(f"  • {tool}: {description}")
        
        # 3. Tool Selection Strategy
        print("\n3. Smart Tool Selection:")
        selection_example = {
            "user_query": "I want something similar to this dress but in red",
            "context_analysis": {
                "intent": "recommend_similar",
                "constraints": {"color": "red"},
                "reference_item": "dress_12345"
            },
            "selected_tool": {
                "tool": "recommend_similar",
                "reasoning": "User wants similar items with color modification",
                "args": {"base_item_id": "dress_12345", "color_filter": "red"}
            },
            "fallback_tools": ["search_items", "visual_search"]
        }
        
        print(json.dumps(selection_example, indent=2))
        
        self.examples["tool_ready"] = {
            "tool_calls": tool_call_example,
            "available_tools": available_tools,
            "selection_strategy": selection_example
        }
    
    def demonstrate_training_pipeline(self):
        """Show training pipeline for SFT, DPO, RL"""
        print("\n🎓 Training Pipeline Support")
        print("="*50)
        
        # 1. SFT (Supervised Fine-Tuning) Data
        print("\n1. SFT Training Examples:")
        sft_example = {
            "context": [
                "User: I need a professional outfit for interviews",
                "Assistant: I'd be happy to help you find professional attire!"
            ],
            "tool_calls": [
                {
                    "tool": "search_items",
                    "args": {
                        "query": "professional interview outfit",
                        "categories": ["blazers", "dresses", "pants"],
                        "style_filters": ["formal", "professional"]
                    }
                }
            ],
            "response": "I found some excellent professional pieces for interviews. This navy blazer would be perfect - it's professional, well-fitted, and versatile for multiple interviews.",
            "intent": "recommend"
        }
        
        print(json.dumps(sft_example, indent=2))
        
        # 2. DPO (Direct Preference Optimization) Data
        print("\n2. DPO Training Pairs:")
        dpo_example = {
            "context": "User looking for red dress under $80",
            "chosen": {
                "tool_calls": [{"tool": "search_items", "args": {"color": "red", "category": "dress", "max_price": 80}}],
                "response": "Here's a beautiful red dress for $75 that fits your budget perfectly!",
                "success_rate": 0.92,
                "user_satisfaction": 0.95
            },
            "rejected": {
                "tool_calls": [{"tool": "filter_items", "args": {"price_range": [0, 1000]}}],
                "response": "Here are some options to consider.",
                "success_rate": 0.45,
                "user_satisfaction": 0.30
            }
        }
        
        print(json.dumps(dpo_example, indent=2))
        
        # 3. RL (Reinforcement Learning) Episodes
        print("\n3. RL Training Episodes:")
        rl_example = {
            "episode_id": "episode_001",
            "steps": [
                {
                    "state": "User: I need a dress for a wedding",
                    "action": {
                        "tool": "search_items",
                        "args": {"query": "wedding guest dress", "occasion": "formal"}
                    },
                    "reward": 0.8,
                    "next_state": "System: I found elegant wedding guest dresses..."
                },
                {
                    "state": "User: Something in blue would be nice",
                    "action": {
                        "tool": "filter_items", 
                        "args": {"color": "blue", "previous_results": True}
                    },
                    "reward": 0.9,
                    "next_state": "System: Here's a stunning blue dress perfect for weddings..."
                }
            ],
            "total_reward": 1.7,
            "success": True,
            "conversation_length": 6
        }
        
        print(json.dumps(rl_example, indent=2))
        
        # 4. Training Pipeline Flow
        print("\n4. Complete Training Flow:")
        training_flow = {
            "stage_1_sft": {
                "input": "Conversation context + correct tool usage",
                "output": "Model learns tool selection and usage patterns",
                "data_format": "context → tool_calls + response"
            },
            "stage_2_dpo": {
                "input": "Chosen vs rejected tool usage pairs",
                "output": "Model learns to prefer better tool selections",
                "data_format": "context → (chosen_response, rejected_response)"
            },
            "stage_3_rl": {
                "input": "Conversation environment simulation",
                "output": "Model optimizes for conversation success",
                "data_format": "state → action → reward → next_state"
            }
        }
        
        print(json.dumps(training_flow, indent=2))
        
        self.examples["training_pipeline"] = {
            "sft_example": sft_example,
            "dpo_example": dpo_example,
            "rl_example": rl_example,
            "training_flow": training_flow
        }
    
    def save_examples(self, filename: str = "muse_v2_examples.json"):
        """Save all examples to file"""
        with open(filename, "w") as f:
            json.dump(self.examples, f, indent=2)
        print(f"\n💾 Examples saved to {filename}")
    
    def generate_comparison_report(self):
        """Generate comparison between original and enhanced MUSe"""
        print("\n📊 MUSe Original vs Enhanced Comparison")
        print("="*60)
        
        comparison = {
            "Data Quality": {
                "Original": "Basic item metadata, simple conversations",
                "Enhanced": "Normalized metadata, natural variations, failure recovery"
            },
            "Context Richness": {
                "Original": "Basic user requirements, simple scenarios",
                "Enhanced": "Rich personas, purchase history, specific goals"
            },
            "Tool Integration": {
                "Original": "No explicit tool usage",
                "Enhanced": "8 tools with alternatives, success tracking"
            },
            "Training Support": {
                "Original": "Basic conversation data",
                "Enhanced": "SFT examples, DPO pairs, RL episodes"
            },
            "Conversation Quality": {
                "Original": "Template-based responses",
                "Enhanced": "Natural variations, intent labeling, persona-driven"
            },
            "Evaluation": {
                "Original": "Basic conversation completion",
                "Enhanced": "Success rate, tool usage, vocabulary diversity"
            }
        }
        
        for category, details in comparison.items():
            print(f"\n{category}:")
            print(f"  Original: {details['Original']}")
            print(f"  Enhanced: {details['Enhanced']}")
        
        return comparison
    
    def run_complete_demonstration(self):
        """Run complete demonstration of all improvements"""
        print("🎉 Enhanced MUSe v2 - Complete Feature Demonstration")
        print("="*70)
        
        # Demonstrate each category
        self.demonstrate_data_quality_improvements()
        self.demonstrate_richer_context()
        self.demonstrate_tool_ready_architecture()
        self.demonstrate_training_pipeline()
        
        # Generate comparison
        comparison = self.generate_comparison_report()
        
        # Save examples
        self.save_examples()
        
        print("\n🎯 Key Benefits Summary:")
        benefits = [
            "🧹 Clean, normalized data ready for training",
            "👤 Rich user personas for personalized conversations",
            "🛠 Explicit tool usage for agentic AI training",
            "📈 Complete training pipeline (SFT → DPO → RL)",
            "💬 Natural dialogue with human-like variations",
            "🔧 Failure simulation and recovery scenarios",
            "📊 Comprehensive evaluation metrics",
            "🔗 Compatible with modern tool-using AI systems"
        ]
        
        for benefit in benefits:
            print(f"  {benefit}")
        
        print("\n🚀 Ready for:")
        applications = [
            "DiaToolDPO training",
            "VisTA multimodal alignment", 
            "ToolFormer tool integration",
            "OctoTools multi-tool usage",
            "Agentic conversation systems",
            "Modern recommendation AI"
        ]
        
        for app in applications:
            print(f"  • {app}")


def main():
    """Main demonstration function"""
    demo = MuseV2Demonstration()
    demo.run_complete_demonstration()


if __name__ == "__main__":
    main()
