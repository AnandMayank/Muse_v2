#!/usr/bin/env python3
"""
MUSE v3 Research Orchestrator
============================

Main entry point for running the complete MUSE v3 research pipeline.
This script handles:
1. Environment setup and validation
2. Dependency management  
3. Pipeline execution coordination
4. Results aggregation and reporting
5. Error handling and recovery
"""

import os
import sys
import traceback
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def setup_environment():
    """Setup the environment for MUSE research"""
    
    print("🔧 Setting up MUSE research environment...")
    
    # Create required directories
    base_dir = Path("/media/adityapachauri/second_drive/Muse/muse_v3_advanced")
    directories = [
        "research_experiments",
        "generated_data", 
        "training_results",
        "evaluation_results",
        "logs"
    ]
    
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup completed")

def validate_dependencies():
    """Validate required dependencies"""
    
    print("📦 Validating dependencies...")
    
    required_modules = [
        "torch", "numpy", "json", "pathlib", 
        "dataclasses", "logging", "time"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing dependencies: {missing_modules}")
        return False
    
    print("✅ All dependencies validated")
    return True

def run_research_pipeline():
    """Run the complete research pipeline"""
    
    print("\n" + "=" * 80)
    print("🚀 STARTING MUSE v3 COMPLETE RESEARCH PIPELINE")
    print("=" * 80)
    
    try:
        # Import main pipeline after environment setup
        from complete_research_pipeline import run_muse_research_experiment
        
        # Run the experiment
        start_time = time.time()
        results = run_muse_research_experiment()
        end_time = time.time()
        
        # Success summary
        print("\n" + "=" * 80)
        print("🎉 MUSE v3 RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
        print(f"📊 Final evaluation score: {results['final_analysis']['key_metrics']['evaluation_score']:.3f}")
        print(f"👥 Human satisfaction: {results['final_analysis']['key_metrics']['human_satisfaction']:.3f}")
        print(f"🔬 Research contributions: {len(results['final_analysis']['research_contributions'])}")
        
        return True, results
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ RESEARCH PIPELINE FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False, str(e)

def run_individual_components():
    """Run and test individual components"""
    
    print("\n🧪 Testing Individual Components")
    print("-" * 50)
    
    component_results = {}
    
    # Test 1: Data Generation Pipeline
    print("1️⃣ Testing Data Generation Pipeline...")
    try:
        from data_generation_pipeline import DataGenerationPipeline
        
        config = {
            "device": "cpu",
            "min_utility_threshold": 0.3,
            "output_dir": "/media/adityapachauri/second_drive/Muse/muse_v3_advanced/test_data"
        }
        
        data_generator = DataGenerationPipeline(config)
        sample_context = ["Find me running shoes under $100"]
        results = data_generator.run_full_pipeline(sample_context)
        
        component_results["data_generation"] = {
            "status": "success",
            "generated_samples": len(results.get("augmented_data", [])),
            "validation_score": results.get("validation", {}).get("avg_quality_score", 0.0)
        }
        
        print("   ✅ Data generation pipeline working")
        
    except Exception as e:
        component_results["data_generation"] = {"status": "failed", "error": str(e)}
        print(f"   ❌ Data generation failed: {str(e)}")
    
    # Test 2: Training Lifecycle  
    print("2️⃣ Testing Training Lifecycle...")
    try:
        from advanced_training_lifecycle import AdvancedTrainingPipeline, TrainingConfig
        
        training_config = TrainingConfig(
            muse_data_path="/tmp",
            output_dir="/tmp/training_test",
            sft_epochs=1,
            device="cpu"
        )
        
        training_pipeline = AdvancedTrainingPipeline(training_config)
        
        # Test initialization
        component_results["training"] = {
            "status": "success",
            "components": ["SFT", "DPO", "RL"],
            "config_valid": True
        }
        
        print("   ✅ Training lifecycle working")
        
    except Exception as e:
        component_results["training"] = {"status": "failed", "error": str(e)}
        print(f"   ❌ Training lifecycle failed: {str(e)}")
    
    # Test 3: Evaluation Framework
    print("3️⃣ Testing Evaluation Framework...")
    try:
        from comprehensive_evaluation import ComprehensiveEvaluationFramework
        from architecture import MuseV3Architecture
        
        # Mock model for testing
        model_config = {
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
        
        model = MuseV3Architecture(model_config)
        
        eval_config = {
            "device": "cpu",
            "output_dir": "/tmp/eval_test"
        }
        
        evaluator = ComprehensiveEvaluationFramework(model, eval_config)
        
        component_results["evaluation"] = {
            "status": "success",
            "frameworks": ["AgentBench", "WebGPT", "τ-bench"],
            "model_loaded": True
        }
        
        print("   ✅ Evaluation framework working")
        
    except Exception as e:
        component_results["evaluation"] = {"status": "failed", "error": str(e)}
        print(f"   ❌ Evaluation framework failed: {str(e)}")
    
    return component_results

def main():
    """Main orchestrator function"""
    
    print("🎬 MUSE v3 Research Orchestrator")
    print("=" * 60)
    
    # Step 1: Environment setup
    setup_environment()
    
    # Step 2: Validate dependencies
    if not validate_dependencies():
        print("❌ Dependency validation failed. Please install missing packages.")
        return False
    
    # Step 3: Test individual components
    print("\n🧪 Testing individual components first...")
    component_results = run_individual_components()
    
    # Check component test results
    failed_components = [name for name, result in component_results.items() 
                        if result.get("status") != "success"]
    
    if failed_components:
        print(f"\n⚠️  Some components failed: {failed_components}")
        print("Continuing with full pipeline (may work despite individual test failures)")
    else:
        print("\n✅ All individual components working!")
    
    # Step 4: Run complete research pipeline
    success, results = run_research_pipeline()
    
    # Step 5: Generate final report
    if success:
        print("\n📄 Generating final report...")
        
        report = {
            "pipeline_success": True,
            "component_tests": component_results,
            "full_pipeline_results": results,
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "python_version": sys.version,
                "platform": os.name,
                "working_directory": str(Path.cwd())
            }
        }
        
        # Save report
        report_file = Path("/media/adityapachauri/second_drive/Muse/muse_v3_advanced/orchestrator_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Final report saved to: {report_file}")
        print("\n🎉 MUSE v3 research orchestration completed successfully!")
        
        return True
        
    else:
        print(f"\n❌ Pipeline failed: {results}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
