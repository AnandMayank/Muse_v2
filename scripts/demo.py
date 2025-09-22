#!/usr/bin/env python3
"""
MUSE v3 Interactive Demo
========================

Interactive demonstration of MUSE v3 capabilities including:
- Multimodal conversation handling
- Cross-lingual support (Hindi-English)
- Dynamic tool orchestration
- Real-time response generation
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add MUSE v3 to path
sys.path.append(str(Path(__file__).parent.parent / "muse_v3_advanced"))

try:
    from muse_v3_system import MuseV3System
    from architecture import MuseV3Architecture
    MUSE_AVAILABLE = True
except ImportError:
    print("⚠️  MUSE v3 components not available. Running in simulation mode.")
    MUSE_AVAILABLE = False

class MuseDemo:
    """Interactive MUSE v3 demonstration"""
    
    def __init__(self):
        self.conversation_history = []
        self.user_profile = {
            "preferences": {},
            "language": "en",
            "session_id": "demo_session_001"
        }
        
        if MUSE_AVAILABLE:
            try:
                self.system = MuseV3System()
                print("✅ MUSE v3 system loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not initialize MUSE v3: {e}")
                print("Running in simulation mode...")
                self.system = None
        else:
            self.system = None
    
    def simulate_response(self, user_input: str) -> Dict[str, Any]:
        """Simulate MUSE response when system is not available"""
        
        # Simple simulation based on input content
        response_data = {
            "response": "",
            "tool_calls": [],
            "intent": "search",
            "language_detected": "en",
            "confidence": 0.85
        }
        
        # Detect language
        if any(char in user_input for char in "हिअआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहक्ष"):
            response_data["language_detected"] = "hi"
        
        # Simulate tool calls based on keywords
        if any(word in user_input.lower() for word in ["find", "search", "ढूंढ", "चाहिए"]):
            response_data["tool_calls"].append({
                "tool_name": "search",
                "arguments": {"query": "extracted from user input"},
                "confidence": 0.9
            })
            
        if any(word in user_input.lower() for word in ["recommend", "suggest", "सुझाव", "बेहतर"]):
            response_data["tool_calls"].append({
                "tool_name": "recommend", 
                "arguments": {"user_preferences": "inferred"},
                "confidence": 0.85
            })
            
        if any(word in user_input.lower() for word in ["compare", "comparison", "तुलना"]):
            response_data["tool_calls"].append({
                "tool_name": "compare",
                "arguments": {"items": ["item1", "item2"]},
                "confidence": 0.88
            })
        
        # Generate appropriate response
        if response_data["language_detected"] == "hi":
            response_data["response"] = f"मैं समझ गया आपकी बात। आपके लिए उपयुक्त विकल्प खोज रहा हूं। {len(response_data['tool_calls'])} tools का उपयोग कर रहा हूं।"
        else:
            response_data["response"] = f"I understand your request. Let me help you find what you're looking for using {len(response_data['tool_calls'])} specialized tools."
        
        return response_data
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate response"""
        
        if self.system:
            try:
                # Use real MUSE system
                response = self.system.process_conversation(
                    user_input=user_input,
                    session_context={
                        "user_profile": self.user_profile,
                        "conversation_history": self.conversation_history[-5:],  # Last 5 turns
                        "language": self.user_profile["language"]
                    }
                )
                return response
                
            except Exception as e:
                print(f"⚠️  Error processing with MUSE system: {e}")
                return self.simulate_response(user_input)
        else:
            # Use simulation
            return self.simulate_response(user_input)
    
    def display_response(self, response_data: Dict[str, Any]):
        """Display formatted response"""
        
        print(f"\n🤖 MUSE v3: {response_data['response']}")
        
        # Show detected language
        lang_name = "Hindi" if response_data["language_detected"] == "hi" else "English"
        print(f"   🗣️  Language: {lang_name}")
        
        # Show tool calls
        if response_data["tool_calls"]:
            print(f"   🛠️  Tools used:")
            for i, tool_call in enumerate(response_data["tool_calls"], 1):
                tool_name = tool_call["tool_name"]
                args = tool_call.get("arguments", {})
                confidence = tool_call.get("confidence", 0.0)
                print(f"      {i}. {tool_name}() - {args} (confidence: {confidence:.2f})")
        
        # Show intent and confidence
        intent = response_data.get("intent", "unknown")
        confidence = response_data.get("confidence", 0.0)
        print(f"   🎯 Intent: {intent} (confidence: {confidence:.2f})")
    
    def run_demo(self):
        """Run interactive demo"""
        
        print("🚀 MUSE v3 Interactive Demo")
        print("=" * 50)
        print("Features demonstrated:")
        print("- Multimodal conversation understanding")
        print("- Cross-lingual support (Hindi-English)")
        print("- Dynamic tool orchestration")
        print("- Context-aware responses")
        print("\nType 'quit' to exit, 'examples' for sample inputs")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for trying MUSE v3!")
                    break
                
                if user_input.lower() == 'examples':
                    self.show_examples()
                    continue
                
                if user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("🗑️  Conversation history cleared")
                    continue
                
                if not user_input:
                    continue
                
                # Process input
                print("\n🤔 Processing...")
                response_data = self.process_input(user_input)
                
                # Display response
                self.display_response(response_data)
                
                # Store in conversation history
                self.conversation_history.append({
                    "user": user_input,
                    "system": response_data["response"],
                    "tool_calls": response_data.get("tool_calls", []),
                    "timestamp": "demo_time"
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def show_examples(self):
        """Show example inputs users can try"""
        
        examples = [
            {
                "category": "🔍 Product Search",
                "examples": [
                    "Find me casual shirts under 2000 rupees",
                    "मुझे programming के लिए laptop चाहिए",
                    "Search for blue formal shirts for office"
                ]
            },
            {
                "category": "🎯 Recommendations", 
                "examples": [
                    "Recommend books for data science beginners",
                    "बच्चों के लिए educational toys suggest करें",
                    "What laptop would you recommend for gaming?"
                ]
            },
            {
                "category": "⚖️ Product Comparison",
                "examples": [
                    "Compare iPhone 15 vs Samsung Galaxy S24",
                    "DELL और HP के laptops में क्या अंतर है",
                    "Show me comparison between these 3 headphones"
                ]
            },
            {
                "category": "🔧 Advanced Queries",
                "examples": [
                    "Find blue shirts, filter by price under 1500, then recommend top 3",
                    "Programming laptop चाहिए 80k budget में with SSD और 16GB RAM",
                    "Search for formal wear, translate descriptions to Hindi"
                ]
            }
        ]
        
        print("\n📋 Example Inputs to Try:")
        print("=" * 30)
        
        for category in examples:
            print(f"\n{category['category']}:")
            for example in category["examples"]:
                print(f"   • {example}")
        
        print(f"\n💡 Tips:")
        print("   • Try mixing Hindi and English")
        print("   • Ask complex multi-step questions") 
        print("   • Use natural conversational language")
        print("   • Type 'clear' to reset conversation")

def main():
    """Main demo function"""
    
    # Check if running from correct directory
    if not Path("muse_v3_advanced").exists():
        print("❌ Please run this demo from the MUSE repository root directory")
        print("Usage: python scripts/demo.py")
        sys.exit(1)
    
    # Initialize and run demo
    demo = MuseDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
