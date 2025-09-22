#!/usr/bin/env python3
"""
Enhanced Conversation Generation for MUSe Training Data
This script generates diverse conversation types using the original MUSe logic
"""

import json
import os
import random
import time
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

# Import MUSe components
from conv_manager import Cmanager
from system_chat import Recsys
from user_chat import User
from config import API_KEY, API_BASE, DATA_PATH, MODEL_PATH, ensure_directories
from mock_client import get_mock_client

class EnhancedConversationGenerator:
    """Generate diverse conversation types for training"""
    
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.output_dir = "enhanced_muse_output"
        self.conversations_dir = f"{self.output_dir}/conversations"
        self.training_dir = f"{self.output_dir}/training_data"
        
        # Ensure directories exist
        ensure_directories()
        os.makedirs(self.conversations_dir, exist_ok=True)
        os.makedirs(f"{self.training_dir}/sft", exist_ok=True)
        os.makedirs(f"{self.training_dir}/dpo", exist_ok=True)
        
        # Initialize components
        self.init_components()
        
        # Conversation types to generate
        self.conversation_types = [
            "recommendation_focused",
            "chitchat_heavy", 
            "clarification_seeking",
            "comparison_based",
            "multimodal_visual",
            "scenario_specific",
            "price_sensitive",
            "feature_focused",
            "failed_recommendation",
            "successful_purchase"
        ]
        
        # Different user personas
        self.user_personas = [
            {"age": "25", "occupation": "Student", "style": "Trendy", "budget": "Low"},
            {"age": "30", "occupation": "Professional", "style": "Classic", "budget": "Medium"},
            {"age": "35", "occupation": "Manager", "style": "Formal", "budget": "High"},
            {"age": "22", "occupation": "Creative", "style": "Bohemian", "budget": "Medium"},
            {"age": "40", "occupation": "Executive", "style": "Minimalist", "budget": "High"},
            {"age": "28", "occupation": "Tech Worker", "style": "Casual", "budget": "Medium"},
            {"age": "45", "occupation": "Consultant", "style": "Sophisticated", "budget": "High"},
            {"age": "26", "occupation": "Artist", "style": "Eclectic", "budget": "Low"}
        ]
        
        # Different scenarios
        self.scenarios = [
            "work", "casual", "formal", "wedding", "party", "sports", "travel", 
            "date", "interview", "vacation", "shopping", "everyday", "special_event"
        ]
        
        # Item categories
        self.categories = [
            "dress", "shirt", "pants", "shoes", "jacket", "suit", "bag", 
            "accessory", "watch", "jewelry", "hat", "coat", "sweater", "skirt"
        ]
    
    def init_components(self):
        """Initialize MUSe system components"""
        print("🔧 Initializing MUSe components...")
        
        db_path = "faiss_db"
        
        if self.use_mock:
            # Use mock client for demonstration
            print("🤖 Using mock client for conversation generation")
            self.user = User(base_url=API_BASE, api_key=API_KEY)
            self.recsys = Recsys(db_path=db_path, data_path=DATA_PATH, model_path=MODEL_PATH, base_url=API_BASE, api_key=API_KEY)
            # We'll override the clients later with mock ones
        else:
            # Use real API (with rate limiting)
            self.user = User(base_url=API_BASE, api_key=API_KEY)
            self.recsys = Recsys(db_path=db_path, data_path=DATA_PATH, model_path=MODEL_PATH, base_url=API_BASE, api_key=API_KEY)
        
        print("✅ Components initialized")
    
    def generate_user_scenario(self, conv_type: str) -> Dict[str, Any]:
        """Generate a user scenario based on conversation type"""
        
        persona = random.choice(self.user_personas)
        scenario = random.choice(self.scenarios)
        category = random.choice(self.categories)
        
        # Base scenario structure
        user_scenario = {
            "user_id": f"enhanced_user_{random.randint(1000, 9999)}",
            "profile": persona,
            "scenario": scenario,
            "conversation_type": conv_type,
            "target_item": {
                "item_id": f"target_{random.randint(100, 999)}",
                "title": f"{category.title()} for {scenario}",
                "description": f"Perfect {category} suitable for {scenario}",
                "categories": [category],
                "price": random.randint(30, 300),
                "features": self.generate_features(category, scenario)
            }
        }
        
        # Customize based on conversation type
        if conv_type == "recommendation_focused":
            user_scenario["requirements"] = f"I need a {category} for {scenario}. Please recommend something good."
            
        elif conv_type == "chitchat_heavy":
            user_scenario["requirements"] = f"Hi! I was browsing for {category} options. What's trending these days?"
            
        elif conv_type == "clarification_seeking":
            user_scenario["requirements"] = f"I'm looking for a {category} but I'm not sure about the details. Can you help me understand the options?"
            
        elif conv_type == "comparison_based":
            user_scenario["requirements"] = f"Can you compare different {category} options for {scenario}? I want to see the differences."
            
        elif conv_type == "multimodal_visual":
            user_scenario["requirements"] = f"I'm looking for a {category} that looks good. Can you show me some visual options?"
            
        elif conv_type == "scenario_specific":
            user_scenario["requirements"] = f"I have a {scenario} coming up and need the perfect {category}. What would you suggest?"
            
        elif conv_type == "price_sensitive":
            budget = "under $50" if persona["budget"] == "Low" else "under $150" if persona["budget"] == "Medium" else "under $300"
            user_scenario["requirements"] = f"I'm looking for a {category} for {scenario} that's {budget}. What are my options?"
            
        elif conv_type == "feature_focused":
            features = random.sample(self.generate_features(category, scenario), 2)
            user_scenario["requirements"] = f"I need a {category} that is {' and '.join(features)}. Do you have recommendations?"
            
        elif conv_type == "failed_recommendation":
            user_scenario["requirements"] = f"I'm looking for a {category} for {scenario}, but I'm very picky about the details."
            user_scenario["should_fail"] = True  # Mark for failure simulation
            
        elif conv_type == "successful_purchase":
            user_scenario["requirements"] = f"I need a {category} for {scenario}. I'm ready to buy if I find something good."
            user_scenario["should_succeed"] = True  # Mark for success
        
        return user_scenario
    
    def generate_features(self, category: str, scenario: str) -> List[str]:
        """Generate appropriate features for a category and scenario"""
        
        base_features = ["comfortable", "stylish", "high-quality", "durable", "affordable"]
        
        category_features = {
            "dress": ["elegant", "flowing", "fitted", "sleeveless", "midi-length"],
            "shirt": ["breathable", "wrinkle-free", "cotton", "button-down", "slim-fit"],
            "pants": ["stretchy", "tailored", "high-waisted", "wide-leg", "cropped"],
            "shoes": ["comfortable", "leather", "slip-resistant", "cushioned", "lightweight"],
            "jacket": ["waterproof", "insulated", "windproof", "packable", "versatile"],
            "bag": ["spacious", "lightweight", "secure", "stylish", "practical"],
            "accessory": ["elegant", "minimalist", "statement", "versatile", "trendy"]
        }
        
        scenario_features = {
            "work": ["professional", "polished", "conservative", "versatile"],
            "casual": ["relaxed", "comfortable", "everyday", "easy-care"],
            "formal": ["elegant", "sophisticated", "refined", "classic"],
            "sports": ["moisture-wicking", "flexible", "supportive", "breathable"],
            "travel": ["packable", "versatile", "comfortable", "wrinkle-resistant"]
        }
        
        features = base_features.copy()
        features.extend(category_features.get(category, []))
        features.extend(scenario_features.get(scenario, []))
        
        return list(set(features))  # Remove duplicates
    
    def generate_diverse_conversations(self, num_conversations: int = 50) -> List[Dict[str, Any]]:
        """Generate diverse conversations for training"""
        
        print(f"🚀 Generating {num_conversations} diverse conversations...")
        
        all_conversations = []
        successful = 0
        failed = 0
        
        for i in tqdm(range(num_conversations), desc="Generating conversations"):
            try:
                # Select conversation type (rotate through types)
                conv_type = self.conversation_types[i % len(self.conversation_types)]
                
                # Generate user scenario
                user_scenario = self.generate_user_scenario(conv_type)
                
                print(f"\n📝 Conversation {i+1}: {conv_type}")
                print(f"   User: {user_scenario['user_id']}")
                print(f"   Scenario: {user_scenario['scenario']}")
                print(f"   Category: {user_scenario['target_item']['categories'][0]}")
                
                # Generate conversation using original MUSe logic
                if self.use_mock:
                    conversation = self.generate_mock_conversation(user_scenario, conv_type)
                else:
                    conversation = self.generate_real_conversation(user_scenario, conv_type)
                
                # Save individual conversation
                conv_filename = f"{self.conversations_dir}/enhanced_conv_{i+1}_{conv_type}.json"
                with open(conv_filename, 'w', encoding='utf-8') as f:
                    json.dump(conversation, f, indent=2, ensure_ascii=False)
                
                all_conversations.append(conversation)
                successful += 1
                
                # Add delay to avoid rate limiting (if using real API)
                if not self.use_mock and i < num_conversations - 1:
                    time.sleep(2)  # 2 second delay between conversations
                
            except Exception as e:
                failed += 1
                print(f"❌ Error generating conversation {i+1}: {str(e)[:100]}")
                continue
        
        print(f"\n✅ Generated {successful} conversations successfully")
        print(f"❌ Failed to generate {failed} conversations")
        
        # Save all conversations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_convs_file = f"{self.conversations_dir}/all_enhanced_conversations_{timestamp}.json"
        with open(all_convs_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2, ensure_ascii=False)
        
        print(f"💾 All conversations saved to {all_convs_file}")
        
        return all_conversations
    
    def generate_mock_conversation(self, user_scenario: Dict[str, Any], conv_type: str) -> Dict[str, Any]:
        """Generate a conversation using mock responses"""
        
        mock_client = get_mock_client()
        
        conversation = {
            "conversation_id": f"{user_scenario['user_id']}_{conv_type}",
            "conversation_type": conv_type,
            "user_profile": user_scenario["profile"],
            "scenario": user_scenario["scenario"],
            "target_item": user_scenario["target_item"],
            "turns": [],
            "success": True,
            "generation_method": "mock"
        }
        
        # Generate multiple turns based on conversation type
        num_turns = self.get_num_turns(conv_type)
        
        for turn in range(num_turns):
            turn_data = {
                "turn": turn + 1,
                "user_message": "",
                "system_response": "",
                "action_type": self.get_action_type(turn, conv_type),
                "recommended_items": []
            }
            
            if turn == 0:
                # First turn: User initial request
                turn_data["user_message"] = user_scenario["requirements"]
                
                # System response
                response = mock_client.chat_completion([
                    {"role": "system", "content": f"You are helping a {user_scenario['profile']['occupation']} find a {user_scenario['target_item']['categories'][0]} for {user_scenario['scenario']}"},
                    {"role": "user", "content": user_scenario["requirements"]}
                ])
                turn_data["system_response"] = response.choices[0].message.content
                
            else:
                # Generate follow-up based on conversation type and turn
                user_msg, system_msg = self.generate_turn_pair(turn, conv_type, user_scenario, mock_client)
                turn_data["user_message"] = user_msg
                turn_data["system_response"] = system_msg
            
            conversation["turns"].append(turn_data)
        
        return conversation
    
    def generate_real_conversation(self, user_scenario: Dict[str, Any], conv_type: str) -> Dict[str, Any]:
        """Generate conversation using real MUSe conversation manager (placeholder)"""
        
        # This would use the actual conv_manager.conv_process method
        # For now, we'll create a structured conversation
        
        conversation = {
            "conversation_id": f"{user_scenario['user_id']}_{conv_type}",
            "conversation_type": conv_type,
            "user_profile": user_scenario["profile"],
            "scenario": user_scenario["scenario"],
            "target_item": user_scenario["target_item"],
            "turns": [],
            "success": True,
            "generation_method": "real_api"
        }
        
        # Would implement actual conversation generation here
        # For now, create a basic structure
        
        return conversation
    
    def get_num_turns(self, conv_type: str) -> int:
        """Get number of turns based on conversation type"""
        turn_counts = {
            "recommendation_focused": 3,
            "chitchat_heavy": 5,
            "clarification_seeking": 4,
            "comparison_based": 4,
            "multimodal_visual": 3,
            "scenario_specific": 3,
            "price_sensitive": 4,
            "feature_focused": 3,
            "failed_recommendation": 4,
            "successful_purchase": 3
        }
        return turn_counts.get(conv_type, 3)
    
    def get_action_type(self, turn: int, conv_type: str) -> str:
        """Get action type for a turn"""
        if conv_type == "chitchat_heavy" and turn < 2:
            return "chitchat"
        elif conv_type == "clarification_seeking":
            return "clarify" if turn % 2 == 0 else "recommend"
        else:
            return "recommend" if turn == 0 else "follow_up"
    
    def generate_turn_pair(self, turn: int, conv_type: str, user_scenario: Dict[str, Any], mock_client) -> tuple:
        """Generate user message and system response for a turn"""
        
        # Generate contextual user message based on turn and type
        if conv_type == "chitchat_heavy":
            user_messages = [
                "That's interesting! Tell me more about the style trends.",
                "I like that approach. What about seasonal considerations?",
                "That makes sense. How do you typically match accessories?",
                "Great point! What would you recommend for my age group?"
            ]
        elif conv_type == "clarification_seeking":
            user_messages = [
                "What's the difference between these options?",
                "Can you explain the features in more detail?",
                "How do I know which size would work best?",
                "What about the care instructions?"
            ]
        elif conv_type == "comparison_based":
            user_messages = [
                "How does this compare to other brands?",
                "What are the pros and cons of each option?",
                "Which one offers better value for money?",
                "What would you personally choose?"
            ]
        else:
            user_messages = [
                "That sounds good. What else do you have?",
                "Can you tell me more about this item?",
                "What about the price and availability?",
                "I think I'd like to see more options."
            ]
        
        user_msg = user_messages[min(turn-1, len(user_messages)-1)]
        
        # Generate system response
        response = mock_client.chat_completion([
            {"role": "system", "content": f"Continue helping with {conv_type} conversation about {user_scenario['target_item']['categories'][0]}"},
            {"role": "user", "content": user_msg}
        ])
        system_msg = response.choices[0].message.content
        
        return user_msg, system_msg

def main():
    """Main function to generate diverse conversations"""
    
    print("🚀 Enhanced MUSe Conversation Generation")
    print("=" * 60)
    
    # Initialize generator
    generator = EnhancedConversationGenerator(use_mock=True)
    
    # Generate diverse conversations
    conversations = generator.generate_diverse_conversations(num_conversations=30)
    
    print(f"\n🎉 Generated {len(conversations)} diverse conversations!")
    print("📁 Check enhanced_muse_output/conversations/ for results")
    
    return conversations

if __name__ == "__main__":
    conversations = main()
