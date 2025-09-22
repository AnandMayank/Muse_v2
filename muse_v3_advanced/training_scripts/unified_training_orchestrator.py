#!/usr/bin/env python3
"""
MUSE v3 Unified Training Orchestrator
=====================================

Full-scale training orchestrator that runs real training scripts:
1. SFT training using sft_training.py
2. DPO training using dpo_training.py  
3. RL training using rl_training.py

This replaces simulated training with actual gradient-based training.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OrchestrationConfig:
    """Configuration for training orchestration"""
    
    # Base paths
    base_dir: str = "/media/adityapachauri/second_drive/Muse"
    training_scripts_dir: str = "muse_v3_advanced/training_scripts"
    data_dir: str = "muse_v3_advanced/research_experiments/muse_v3_comprehensive_study/generated_data"
    output_dir: str = "muse_v3_advanced/training_outputs"
    
    # Training stages
    run_sft: bool = True
    run_dpo: bool = True
    run_rl: bool = True
    
    # Training parameters
    sft_epochs: int = 3
    dpo_epochs: int = 2
    rl_episodes: int = 500
    
    # System parameters
    device: str = "cpu"
    max_memory_gb: int = 8
    parallel_processes: int = 1
    
    # Monitoring
    log_interval: int = 10
    save_interval: int = 100

# =============================================================================
# TRAINING ORCHESTRATOR
# =============================================================================

class UnifiedTrainingOrchestrator:
    """
    Unified orchestrator for real MUSE training pipeline
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.base_path = Path(config.base_dir)
        self.training_scripts_path = self.base_path / config.training_scripts_dir
        self.data_path = self.base_path / config.data_dir
        self.output_path = self.base_path / config.output_dir
        
        # Create output directories
        self._setup_directories()
        
        # Training state
        self.training_results = {}
        self.start_time = time.time()
        
        logger.info("🎯 Unified Training Orchestrator initialized")
    
    def _setup_directories(self):
        """Setup required directories"""
        directories = [
            self.output_path / "sft",
            self.output_path / "sft" / "checkpoints",
            self.output_path / "dpo", 
            self.output_path / "dpo" / "checkpoints",
            self.output_path / "rl",
            self.output_path / "rl" / "checkpoints",
            self.output_path / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("📁 Created output directories")
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline with real training scripts"""
        
        logger.info("🚀 Starting MUSE v3 Full-Scale Training Pipeline")
        print("=" * 80)
        
        pipeline_results = {
            "start_time": time.time(),
            "stages": {},
            "overall_success": True,
            "total_duration": 0
        }
        
        try:
            # Stage 1: Supervised Fine-Tuning (SFT)
            if self.config.run_sft:
                logger.info("🔥 Stage 1: Running SFT Training")
                sft_results = self._run_sft_training()
                pipeline_results["stages"]["sft"] = sft_results
                
                if not sft_results.get("success", False):
                    logger.error("❌ SFT training failed, stopping pipeline")
                    pipeline_results["overall_success"] = False
                    return pipeline_results
            
            # Stage 2: Direct Preference Optimization (DPO)
            if self.config.run_dpo and pipeline_results["overall_success"]:
                logger.info("🎯 Stage 2: Running DPO Training")
                dpo_results = self._run_dpo_training()
                pipeline_results["stages"]["dpo"] = dpo_results
                
                if not dpo_results.get("success", False):
                    logger.error("❌ DPO training failed, stopping pipeline")
                    pipeline_results["overall_success"] = False
                    return pipeline_results
            
            # Stage 3: Reinforcement Learning (RL)
            if self.config.run_rl and pipeline_results["overall_success"]:
                logger.info("🎮 Stage 3: Running RL Training")
                rl_results = self._run_rl_training()
                pipeline_results["stages"]["rl"] = rl_results
                
                if not rl_results.get("success", False):
                    logger.error("❌ RL training failed")
                    pipeline_results["overall_success"] = False
        
        except Exception as e:
            logger.error(f"💥 Pipeline failed with error: {str(e)}")
            pipeline_results["overall_success"] = False
            pipeline_results["error"] = str(e)
        
        # Finalize results
        pipeline_results["end_time"] = time.time()
        pipeline_results["total_duration"] = pipeline_results["end_time"] - pipeline_results["start_time"]
        
        self._log_pipeline_summary(pipeline_results)
        self._save_training_logs(pipeline_results)
        
        return pipeline_results
    
    def _run_sft_training(self) -> Dict[str, Any]:
        """Run SFT training using real training script"""
        
        stage_start = time.time()
        logger.info("🔥 Executing SFT Training Script...")
        
        # Prepare SFT command
        sft_script = self.training_scripts_path / "sft_training.py"
        
        # Check if script exists
        if not sft_script.exists():
            return {
                "success": False,
                "error": f"SFT script not found: {sft_script}",
                "duration": 0
            }
        
        # Prepare environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.base_path)
        
        try:
            # Run SFT training
            cmd = [
                sys.executable,
                str(sft_script)
            ]
            
            logger.info(f"🔧 Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.base_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - stage_start
            
            if result.returncode == 0:
                logger.info("✅ SFT training completed successfully")
                return {
                    "success": True,
                    "duration": duration,
                    "stdout": result.stdout,
                    "final_checkpoint": str(self.output_path / "sft" / "checkpoints" / "best_sft_model.pt")
                }
            else:
                logger.error(f"❌ SFT training failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "duration": duration,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "return_code": result.returncode
                }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "duration": time.time() - stage_start,
                "error": "SFT training timeout (1 hour)"
            }
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - stage_start,
                "error": f"SFT training exception: {str(e)}"
            }
    
    def _run_dpo_training(self) -> Dict[str, Any]:
        """Run DPO training using real training script"""
        
        stage_start = time.time()
        logger.info("🎯 Executing DPO Training Script...")
        
        # Prepare DPO command
        dpo_script = self.training_scripts_path / "dpo_training.py"
        
        # Check if script exists
        if not dpo_script.exists():
            return {
                "success": False,
                "error": f"DPO script not found: {dpo_script}",
                "duration": 0
            }
        
        # Check if SFT checkpoint exists
        sft_checkpoint = self.output_path / "sft" / "checkpoints" / "best_sft_model.pt"
        if not sft_checkpoint.exists():
            logger.warning("⚠️ SFT checkpoint not found, DPO will use random initialization")
        
        # Prepare environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.base_path)
        
        try:
            # Run DPO training
            cmd = [
                sys.executable,
                str(dpo_script)
            ]
            
            logger.info(f"🔧 Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.base_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - stage_start
            
            if result.returncode == 0:
                logger.info("✅ DPO training completed successfully")
                return {
                    "success": True,
                    "duration": duration,
                    "stdout": result.stdout,
                    "final_checkpoint": str(self.output_path / "dpo" / "checkpoints" / "best_dpo_model.pt")
                }
            else:
                logger.error(f"❌ DPO training failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "duration": duration,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "return_code": result.returncode
                }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "duration": time.time() - stage_start,
                "error": "DPO training timeout (1 hour)"
            }
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - stage_start,
                "error": f"DPO training exception: {str(e)}"
            }
    
    def _run_rl_training(self) -> Dict[str, Any]:
        """Run RL training using real training script"""
        
        stage_start = time.time()
        logger.info("🎮 Executing RL Training Script...")
        
        # Prepare RL command
        rl_script = self.training_scripts_path / "rl_training.py"
        
        # Check if script exists
        if not rl_script.exists():
            return {
                "success": False,
                "error": f"RL script not found: {rl_script}",
                "duration": 0
            }
        
        # Check if DPO checkpoint exists
        dpo_checkpoint = self.output_path / "dpo" / "checkpoints" / "best_dpo_model.pt"
        if not dpo_checkpoint.exists():
            logger.warning("⚠️ DPO checkpoint not found, RL will use random initialization")
        
        # Prepare environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.base_path)
        
        try:
            # Run RL training
            cmd = [
                sys.executable,
                str(rl_script)
            ]
            
            logger.info(f"🔧 Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.base_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout for RL
            )
            
            duration = time.time() - stage_start
            
            if result.returncode == 0:
                logger.info("✅ RL training completed successfully")
                return {
                    "success": True,
                    "duration": duration,
                    "stdout": result.stdout,
                    "final_checkpoint": str(self.output_path / "rl" / "checkpoints" / "best_rl_model.pt")
                }
            else:
                logger.error(f"❌ RL training failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "duration": duration,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "return_code": result.returncode
                }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "duration": time.time() - stage_start,
                "error": "RL training timeout (2 hours)"
            }
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - stage_start,
                "error": f"RL training exception: {str(e)}"
            }
    
    def _log_pipeline_summary(self, results: Dict[str, Any]):
        """Log comprehensive pipeline summary"""
        
        print("\n" + "=" * 80)
        print("🎉 MUSE v3 TRAINING PIPELINE SUMMARY")
        print("=" * 80)
        
        # Overall status
        status = "✅ SUCCESS" if results["overall_success"] else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Total Duration: {results['total_duration']:.2f} seconds")
        print(f"Start Time: {time.ctime(results['start_time'])}")
        print(f"End Time: {time.ctime(results['end_time'])}")
        
        # Stage-by-stage results
        print("\n📊 STAGE RESULTS:")
        print("-" * 40)
        
        stages = results.get("stages", {})
        
        for stage_name, stage_results in stages.items():
            stage_status = "✅" if stage_results.get("success", False) else "❌"
            duration = stage_results.get("duration", 0)
            
            print(f"{stage_name.upper()}: {stage_status} ({duration:.2f}s)")
            
            if not stage_results.get("success", False):
                error = stage_results.get("error", "Unknown error")
                print(f"  Error: {error}")
            else:
                checkpoint = stage_results.get("final_checkpoint")
                if checkpoint:
                    print(f"  Checkpoint: {checkpoint}")
        
        # Final model path
        if results["overall_success"]:
            final_model = self.output_path / "rl" / "checkpoints" / "best_rl_model.pt"
            print(f"\n🏆 FINAL MODEL: {final_model}")
        
        print("=" * 80)
    
    def _save_training_logs(self, results: Dict[str, Any]):
        """Save training logs to file"""
        
        log_file = self.output_path / "logs" / "full_training_log.json"
        
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Training logs saved to: {log_file}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        
        status = {
            "orchestrator_running": True,
            "elapsed_time": time.time() - self.start_time,
            "stages_completed": list(self.training_results.keys()),
            "current_stage": None
        }
        
        return status

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main orchestration script"""
    
    print("🎯 MUSE v3 Full-Scale Training Orchestrator")
    print("=" * 60)
    
    # Configuration
    config = OrchestrationConfig(
        run_sft=True,
        run_dpo=True, 
        run_rl=True,
        sft_epochs=2,  # Reduced for testing
        dpo_epochs=2,
        rl_episodes=200,  # Reduced for testing
        device="cpu"
    )
    
    print(f"📋 Configuration:")
    print(f"  - Base Directory: {config.base_dir}")
    print(f"  - Training Stages: SFT={config.run_sft}, DPO={config.run_dpo}, RL={config.run_rl}")
    print(f"  - Device: {config.device}")
    
    # Initialize orchestrator
    orchestrator = UnifiedTrainingOrchestrator(config)
    
    # Run full training pipeline
    print("\n🚀 Starting full-scale training pipeline...")
    results = orchestrator.run_full_training_pipeline()
    
    # Final status
    if results["overall_success"]:
        print("\n🎉 MUSE v3 training pipeline completed successfully!")
        print(f"🏆 Final trained model available at: {config.base_dir}/muse_v3_advanced/training_outputs/rl/checkpoints/")
    else:
        print("\n💥 MUSE v3 training pipeline failed!")
        print("🔍 Check logs for details.")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)
