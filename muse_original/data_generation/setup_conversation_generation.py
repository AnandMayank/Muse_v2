#!/usr/bin/env python3
"""
Setup script for Enhanced MUSe Conversation Generation
This script will create the necessary files and prepare the system for conversation generation.
"""

import json
import os
import shutil
from pathlib import Path
from config import ensure_directories, validate_config

def create_user_scenarios():
    """Create user_scenarios_7005.json from sample_scenarios.json or generate new ones"""
    
    print("📝 Creating user scenarios file...")
    
    # Check if sample_scenarios.json exists
    if os.path.exists('sample_scenarios.json'):
        print("✅ Found sample_scenarios.json, copying to user_scenarios_7005.json")
        shutil.copy('sample_scenarios.json', 'user_scenarios_7005.json')
        return True
    
    # Check if scenarios.json exists
    elif os.path.exists('scenarios.json'):
        print("✅ Found scenarios.json, copying to user_scenarios_7005.json")
        shutil.copy('scenarios.json', 'user_scenarios_7005.json')
        return True
    
    # Generate basic scenarios
    else:
        print("🎯 Generating basic user scenarios...")
        basic_scenarios = []
        
        scenarios = ['work', 'casual', 'wedding', 'sports', 'travel', 'party', 'date', 'vacation']
        items = ['dress', 'shirt', 'pants', 'shoes', 'jacket', 'suit', 'bag', 'accessory']
        
        for i in range(50):  # Generate 50 scenarios
            scenario = scenarios[i % len(scenarios)]
            item = items[i % len(items)]
            age = 20 + (i % 50)
            price = 50 + (i % 200)
            
            user_scenario = {
                "user_id": f"user_{i:03d}",
                "profile": {
                    "age": str(age),
                    "occupation": "Professional" if i % 2 == 0 else "Student",
                    "style": "Modern" if i % 3 == 0 else "Classic"
                },
                "scenario": scenario,
                "requirements": f"Looking for a {item} for {scenario}",
                "target_item": {
                    "item_id": f"target_{i:03d}",
                    "title": f"Perfect {item}",
                    "description": f"Ideal {item} for {scenario}",
                    "categories": [item],
                    "price": price,
                    "features": ["comfortable", "stylish", "high-quality"]
                }
            }
            basic_scenarios.append(user_scenario)
        
        # Save to file
        with open('user_scenarios_7005.json', 'w', encoding='utf-8') as f:
            json.dump(basic_scenarios, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Generated {len(basic_scenarios)} user scenarios")
        return True

def create_basic_item_profile():
    """Create a basic updated_item_profile.json if it doesn't exist"""
    
    if os.path.exists('updated_item_profile.json'):
        print("✅ updated_item_profile.json already exists")
        return True
    
    print("📦 Creating basic item profile...")
    
    # Check if there's an existing item profile to copy
    profile_files = ['item_profile.json', 'sample_item_profile.json']
    for profile_file in profile_files:
        if os.path.exists(profile_file):
            print(f"✅ Found {profile_file}, copying to updated_item_profile.json")
            shutil.copy(profile_file, 'updated_item_profile.json')
            return True
    
    # Generate basic item profile
    print("🎯 Generating basic item profile...")
    basic_items = {}
    
    categories = ['dress', 'shirt', 'pants', 'shoes', 'jacket', 'suit', 'bag', 'accessory']
    
    for i in range(100):  # Generate 100 items
        category = categories[i % len(categories)]
        price = 20 + (i % 300)
        
        item = {
            f"item_{i:03d}": {
                "title": f"{category.title()} Item {i:03d}",
                "description": f"High-quality {category} with modern design and comfortable fit",
                "categories": [category],
                "price": price,
                "features": ["comfortable", "durable", "stylish"],
                "new_description": f"This {category} features excellent craftsmanship with attention to detail. Perfect for various occasions with its versatile design and premium materials."
            }
        }
        basic_items.update(item)
    
    # Save to file
    with open('updated_item_profile.json', 'w', encoding='utf-8') as f:
        json.dump(basic_items, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(basic_items)} items in profile")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'openai', 'tqdm', 'requests', 'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_faiss_directory():
    """Create FAISS database directory"""
    
    faiss_dir = "faiss_db"
    if not os.path.exists(faiss_dir):
        print(f"📁 Creating {faiss_dir} directory...")
        os.makedirs(faiss_dir, exist_ok=True)
        print("✅ FAISS directory created")
    else:
        print("✅ FAISS directory already exists")
    
    return True

def main():
    """Main setup function"""
    
    print("🚀 Enhanced MUSe Setup Script")
    print("=" * 50)
    
    # Validate configuration
    print("1️⃣ Validating configuration...")
    if not validate_config():
        print("❌ Please fix configuration issues first")
        return False
    
    # Ensure directories
    print("\n2️⃣ Creating directories...")
    ensure_directories()
    create_faiss_directory()
    
    # Check dependencies
    print("\n3️⃣ Checking dependencies...")
    if not check_dependencies():
        print("❌ Please install missing packages first")
        return False
    
    # Create required files
    print("\n4️⃣ Creating required files...")
    create_user_scenarios()
    create_basic_item_profile()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Check the enhanced_muse_output/ directory for results")
    print("3. Review logs/ directory for detailed logs")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        exit(1)
    else:
        print("\n✅ Ready to generate conversations!")
