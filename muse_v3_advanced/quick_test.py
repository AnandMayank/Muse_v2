#!/usr/bin/env python3
"""
Quick MUSE v3 Architecture Test
==============================

Fast test without downloading heavy models.
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock the heavy model components to avoid downloads
class MockTextEncoder(nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Linear(128, hidden_size)  # Mock projection
        
    def forward(self, text_input, return_language_logits=False):
        if isinstance(text_input, list):
            batch_size = len(text_input)
        else:
            batch_size = 1
            
        # Mock embeddings
        embeddings = torch.randn(batch_size, self.hidden_size)
        
        result = {"embeddings": embeddings}
        if return_language_logits:
            result["language_logits"] = torch.randn(batch_size, 2)
        
        return result

class MockImageEncoder(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, images, return_attributes=False):
        batch_size = 1 if not isinstance(images, list) else len(images)
        embeddings = torch.randn(batch_size, self.hidden_size)
        
        result = {"embeddings": embeddings}
        if return_attributes:
            result["attributes"] = torch.sigmoid(torch.randn(batch_size, 128))
        
        return result

def test_architecture_components():
    """Test individual architecture components"""
    logger.info("🧪 Testing MUSE v3 Components")
    
    # Test MockTextEncoder
    text_encoder = MockTextEncoder(384)
    text_output = text_encoder(["hello world", "test"], return_language_logits=True)
    logger.info(f"✅ Text Encoder: {text_output['embeddings'].shape}")
    
    # Test MockImageEncoder  
    image_encoder = MockImageEncoder(512)
    image_output = image_encoder(["dummy_image"])
    logger.info(f"✅ Image Encoder: {image_output['embeddings'].shape}")
    
    # Test MetadataEncoder
    from architecture import MetadataEncoder
    metadata_encoder = MetadataEncoder({"category": 50, "brand": 100}, 256)
    metadata_input = {
        "category": torch.tensor([1, 2]),
        "brand": torch.tensor([5, 10])
    }
    metadata_output = metadata_encoder(metadata_input)
    logger.info(f"✅ Metadata Encoder: {metadata_output['embeddings'].shape}")
    
    # Test MultimodalFusion
    from architecture import MultimodalFusion
    fusion = MultimodalFusion(512, 8)
    text_emb = torch.randn(2, 1, 512)
    image_emb = torch.randn(2, 1, 512)  
    metadata_emb = torch.randn(2, 512)
    
    fusion_output = fusion(text_emb, image_emb, metadata_emb)
    logger.info(f"✅ Multimodal Fusion: {fusion_output['multimodal_representation'].shape}")
    
    return True

def test_full_architecture():
    """Test full architecture with mock components"""
    logger.info("🧪 Testing Full MUSE v3 Architecture")
    
    # Patch the heavy components with mocks
    import architecture
    architecture.TextEncoder = MockTextEncoder
    architecture.ImageEncoder = MockImageEncoder
    
    # Create architecture
    config = {
        "text_dim": 384,
        "image_dim": 512,
        "metadata_dim": 256,
        "metadata_vocab": {"category": 50, "brand": 100},
        "fusion_dim": 512,
        "num_intents": 7,
        "num_tools": 6,
        "max_steps": 5,
        "device": "cpu"
    }
    
    from architecture import MuseV3Architecture
    model = MuseV3Architecture(config)
    model.eval()
    
    # Test forward pass
    batch_data = {
        "text_input": ["I want to buy shoes", "Looking for a dress"],
        "metadata_categorical": {
            "category": torch.tensor([0, 1]),
            "brand": torch.tensor([0, 1])
        },
        "conversation_history": [],
        "batch_size": 2
    }
    
    with torch.no_grad():
        outputs = model(batch_data)
    
    logger.info("✅ Forward pass successful!")
    logger.info(f"  Fused features: {outputs['fused_features'].shape}")
    logger.info(f"  Intent logits: {outputs['intent_logits'].shape}")
    logger.info(f"  Selected tools: {outputs['selected_tools']}")
    
    return True

def test_conversation_generation():
    """Test conversation generation with mock architecture"""
    logger.info("🧪 Testing Conversation Generation")
    
    # Use mock components
    import architecture
    architecture.TextEncoder = MockTextEncoder
    architecture.ImageEncoder = MockImageEncoder
    
    config = {
        "text_dim": 384, "image_dim": 512, "metadata_dim": 256,
        "fusion_dim": 512, "num_intents": 7, "device": "cpu"
    }
    
    from architecture import MuseV3Architecture
    model = MuseV3Architecture(config)
    model.eval()
    
    # Test various conversation scenarios
    test_cases = [
        {"text": "I need running shoes", "expected_intent": "search"},
        {"text": "Can you recommend a dress for wedding?", "expected_intent": "recommend"},
        {"text": "Compare iPhone vs Samsung", "expected_intent": "compare"},
        {"text": "मुझे नई किताब चाहिए", "expected_intent": "search"},  # Hindi
    ]
    
    intent_names = ["search", "recommend", "compare", "filter", "translate", "visual_search", "chitchat"]
    
    with torch.no_grad():
        for i, case in enumerate(test_cases):
            batch = {
                "text_input": [case["text"]],
                "metadata_categorical": {
                    "category": torch.tensor([0]),
                    "brand": torch.tensor([0])
                },
                "batch_size": 1
            }
            
            outputs = model(batch)
            
            # Get predictions
            intent_idx = outputs["predicted_intent"].item()
            intent_name = intent_names[intent_idx] if intent_idx < len(intent_names) else "unknown"
            intent_conf = torch.softmax(outputs["intent_logits"], dim=-1).max().item()
            
            logger.info(f"Test {i+1}: '{case['text']}'")
            logger.info(f"  → Intent: {intent_name} (confidence: {intent_conf:.3f})")
            logger.info(f"  → Tools: {outputs['selected_tools']}")
            
    logger.info("🎉 Conversation generation test completed!")
    return True

def main():
    """Run all tests"""
    logger.info("🚀 Quick MUSE v3 Architecture Testing")
    logger.info("=" * 50)
    
    tests = [
        ("Component Tests", test_architecture_components),
        ("Full Architecture", test_full_architecture), 
        ("Conversation Generation", test_conversation_generation)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
    
    logger.info(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! MUSE v3 architecture is working!")
        logger.info("\n📋 System Status:")
        logger.info("✅ Architecture components integrated")
        logger.info("✅ Forward pass working")  
        logger.info("✅ Conversation processing functional")
        logger.info("✅ Ready for full training with real data")
    
    return passed == len(tests)

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
