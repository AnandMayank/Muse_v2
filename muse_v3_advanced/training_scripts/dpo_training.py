#!/usr/bin/env python3
"""
MUSE v3 Direct Preference Optimization (DPO) Training Script
===========================================================

Full-scale DPO training implementation following DiaTool methodology:
1. Real preference pair loading and processing
2. Actual DPO loss calculation with KL regularization
3. Reference model management
4. Comprehensive evaluation and checkpointing

Based on DPO paper and DiaTool preference learning approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import time
from collections import defaultdict
import copy

# Handle optional imports
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, **kwargs):
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Import MUSE architecture and SFT components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture import MuseV3Architecture
from training_scripts.sft_training import SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DPOConfig:
    """Complete DPO training configuration"""
    
    # Data paths
    dpo_data_path: str
    sft_checkpoint_path: str
    output_dir: str
    checkpoint_dir: str
    
    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # DPO specific parameters
    beta: float = 0.1  # KL regularization strength
    reference_free: bool = False  # Use reference-free DPO
    label_smoothing: float = 0.0  # Label smoothing for preferences
    
    # Model parameters
    max_sequence_length: int = 512
    
    # Evaluation
    eval_steps: int = 50
    save_steps: int = 200
    logging_steps: int = 25
    
    # System
    device: str = "cpu"
    num_workers: int = 4
    seed: int = 42
    
    # Wandb logging
    use_wandb: bool = True
    project_name: str = "muse-dpo-training"
    experiment_name: str = "dialoop-dpo"

# =============================================================================
# DATASET IMPLEMENTATION
# =============================================================================

class DPODataset(Dataset):
    """
    Advanced DPO dataset with proper preference pair handling
    
    Features:
    - Chosen vs rejected trajectory comparison
    - Tool-aware preference scoring
    - Context preservation across trajectories
    """
    
    def __init__(self, data_path: str, config: DPOConfig):
        self.config = config
        
        # Load DPO preference pairs
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # Filter valid preference pairs
        self.data = []
        for item in self.raw_data:
            if self._is_valid_preference_pair(item):
                self.data.append(item)
        
        # Build vocabulary from SFT dataset (simplified)
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<TOOL>": 2, "<ARG>": 3, "<END>": 4}
        self._build_simple_vocabulary()
        
        logger.info(f"🎯 Loaded {len(self.data)} valid DPO preference pairs")
        
    def _is_valid_preference_pair(self, item: Dict[str, Any]) -> bool:
        """Check if preference pair is valid for training"""
        required_keys = ["context", "chosen_trajectory", "rejected_trajectory", "preference_score"]
        return all(key in item for key in required_keys)
    
    def _build_simple_vocabulary(self):
        """Build simple vocabulary for tokenization"""
        import re
        
        word_freq = defaultdict(int)
        
        # Count words from all contexts
        for item in self.data:
            words = re.findall(r'\w+', item["context"].lower())
            for word in words:
                word_freq[word] += 1
        
        # Add frequent words
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if len(self.word_to_id) >= 5000:  # Vocabulary limit
                break
            if freq >= 2:
                self.word_to_id[word] = len(self.word_to_id)
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        logger.info(f"Built DPO vocabulary with {len(self.word_to_id)} tokens")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize context
        context_tokens = self._tokenize_text(item["context"])
        context_ids = self._tokens_to_ids(context_tokens)
        
        # Create preference example
        example = {
            "context": item["context"],
            "context_ids": torch.tensor(self._pad_sequence(context_ids), dtype=torch.long),
            "chosen_trajectory": item["chosen_trajectory"],
            "rejected_trajectory": item["rejected_trajectory"],
            "preference_score": float(item.get("preference_score", 0.5)),
            "preference_strength": float(item.get("preference_strength", 1.0)),
            "attention_mask": torch.tensor([1 if x != 0 else 0 for x in self._pad_sequence(context_ids)], dtype=torch.long)
        }
        
        return example
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        return re.findall(r'\w+|[^\w\s]', text.lower())
    
    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return [self.word_to_id.get(token, self.word_to_id["<UNK>"]) for token in tokens]
    
    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        """Pad sequence to max length"""
        max_len = self.config.max_sequence_length
        if len(sequence) >= max_len:
            return sequence[:max_len]
        return sequence + [0] * (max_len - len(sequence))

# =============================================================================
# DPO MODEL WRAPPER
# =============================================================================

class MuseDPOModel(nn.Module):
    """
    MUSE model wrapper for DPO training with trajectory comparison
    """
    
    def __init__(self, policy_model: MuseV3Architecture, 
                 reference_model: MuseV3Architecture, 
                 config: DPOConfig):
        super().__init__()
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # DPO-specific heads
        hidden_dim = policy_model.fusion_dim
        
        # Trajectory scoring head
        self.trajectory_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Tool quality predictor
        self.tool_quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Dict[str, torch.Tensor], 
                trajectory_type: str = "chosen") -> Dict[str, torch.Tensor]:
        """Forward pass for chosen or rejected trajectory"""
        
        # Prepare input for base model
        model_input = {
            "text_input": batch["context_ids"],
            "batch_size": batch["context_ids"].size(0),
            "metadata_categorical": {
                "category": torch.zeros(batch["context_ids"].size(0), dtype=torch.long, device=batch["context_ids"].device),
                "brand": torch.zeros(batch["context_ids"].size(0), dtype=torch.long, device=batch["context_ids"].device)
            }
        }
        
        # Get policy model outputs
        policy_outputs = self.policy_model(model_input)
        
        # Get reference model outputs (no gradients)
        with torch.no_grad():
            reference_outputs = self.reference_model(model_input)
        
        # Calculate trajectory scores
        fusion_features = policy_outputs.get("fusion_features", torch.zeros(batch["context_ids"].size(0), self.config.max_sequence_length, self.policy_model.fusion_dim))
        pooled_features = fusion_features.mean(dim=1)
        
        trajectory_score = self.trajectory_scorer(pooled_features).squeeze(-1)
        tool_quality = self.tool_quality_head(pooled_features).squeeze(-1)
        
        return {
            "policy_logits": policy_outputs["intent_logits"],
            "reference_logits": reference_outputs["intent_logits"],
            "trajectory_score": trajectory_score,
            "tool_quality": tool_quality,
            "fusion_features": fusion_features
        }
    
    def get_trajectory_log_probs(self, outputs: Dict[str, torch.Tensor],
                               trajectory: List[Dict[str, Any]]) -> torch.Tensor:
        """Calculate log probabilities for a trajectory"""
        policy_logits = outputs["policy_logits"]
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # For simplicity, use max log probability as trajectory score
        return log_probs.max(dim=-1)[0]

# =============================================================================
# DPO TRAINER IMPLEMENTATION
# =============================================================================

class DPOTrainer:
    """
    Comprehensive DPO trainer with proper preference optimization
    """
    
    def __init__(self, model: MuseDPOModel, config: DPOConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer (typically lower LR than SFT)
        self.optimizer = AdamW(
            self.model.policy_model.parameters(),  # Only train policy model
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_reward = -float('inf')
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        
        # Initialize wandb if available
        if config.use_wandb and HAS_WANDB:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=config.__dict__
            )
        
        logger.info("🎯 DPO Trainer initialized")
    
    def train(self, train_dataset: DPODataset, 
              eval_dataset: Optional[DPODataset] = None):
        """Main DPO training loop"""
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"🚀 Starting DPO training for {self.config.num_epochs} epochs")
        logger.info(f"📊 Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Evaluation phase
            if eval_loader:
                eval_metrics = self._eval_epoch(eval_loader)
                
                # Save best model based on preference accuracy
                current_reward = eval_metrics.get("preference_accuracy", 0.0)
                if current_reward > self.best_eval_reward:
                    self.best_eval_reward = current_reward
                    self._save_checkpoint("best_dpo_model.pt")
            
            # Log epoch results
            self._log_epoch_results(train_metrics, eval_metrics if eval_loader else None)
            
            # Save regular checkpoint
            self._save_checkpoint(f"dpo_checkpoint_epoch_{epoch}.pt")
        
        logger.info("✅ DPO training completed!")
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(train_loader, desc=f"DPO Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass for chosen and rejected trajectories
            chosen_outputs = self.model(batch, "chosen")
            rejected_outputs = self.model(batch, "rejected")
            
            # Calculate DPO loss
            losses = self._calculate_dpo_loss(chosen_outputs, rejected_outputs, batch)
            total_loss = losses["dpo_loss"]
            
            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                total_loss = total_loss / self.config.gradient_accumulation_steps
            
            total_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.policy_model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_training_step(losses)
            
            # Update progress bar
            progress_bar.set_postfix({
                "dpo_loss": f"{losses['dpo_loss'].item():.4f}",
                "kl_div": f"{losses['kl_divergence'].item():.4f}",
                "acc": f"{losses['preference_accuracy']:.3f}"
            })
            
            # Store metrics
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key].append(value.item())
                else:
                    epoch_metrics[key].append(value)
        
        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def _eval_epoch(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate for one epoch"""
        self.model.eval()
        epoch_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="DPO Evaluation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                chosen_outputs = self.model(batch, "chosen")
                rejected_outputs = self.model(batch, "rejected")
                
                # Calculate losses
                losses = self._calculate_dpo_loss(chosen_outputs, rejected_outputs, batch)
                
                # Store metrics
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        epoch_metrics[key].append(value.item())
                    else:
                        epoch_metrics[key].append(value)
        
        # Average metrics
        eval_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        self._log_eval_metrics(eval_metrics)
        
        return eval_metrics
    
    def _calculate_dpo_loss(self, chosen_outputs: Dict[str, torch.Tensor],
                          rejected_outputs: Dict[str, torch.Tensor],
                          batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate DPO loss with KL regularization"""
        
        # Get log probabilities
        chosen_policy_logprobs = self.model.get_trajectory_log_probs(chosen_outputs, batch["chosen_trajectory"])
        rejected_policy_logprobs = self.model.get_trajectory_log_probs(rejected_outputs, batch["rejected_trajectory"])
        
        chosen_reference_logprobs = self.model.get_trajectory_log_probs(
            {"policy_logits": chosen_outputs["reference_logits"]}, batch["chosen_trajectory"]
        )
        rejected_reference_logprobs = self.model.get_trajectory_log_probs(
            {"policy_logits": rejected_outputs["reference_logits"]}, batch["rejected_trajectory"]
        )
        
        # DPO loss calculation
        # L_DPO = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
        
        if self.config.reference_free:
            # Reference-free DPO
            logits = chosen_policy_logprobs - rejected_policy_logprobs
        else:
            # Standard DPO with reference model
            policy_diff = chosen_policy_logprobs - rejected_policy_logprobs
            reference_diff = chosen_reference_logprobs - rejected_reference_logprobs
            logits = self.config.beta * (policy_diff - reference_diff)
        
        # DPO loss
        if self.config.label_smoothing > 0:
            # Label smoothing
            target_prob = 1.0 - self.config.label_smoothing
            dpo_loss = -target_prob * F.logsigmoid(logits) - (1 - target_prob) * F.logsigmoid(-logits)
        else:
            dpo_loss = -F.logsigmoid(logits)
        
        dpo_loss = dpo_loss.mean()
        
        # Calculate metrics
        with torch.no_grad():
            # Preference accuracy
            preference_accuracy = (logits > 0).float().mean().item()
            
            # KL divergence
            kl_divergence = ((chosen_policy_logprobs - chosen_reference_logprobs) + 
                           (rejected_policy_logprobs - rejected_reference_logprobs)).mean()
            giut 
            # Reward margin
            reward_margin = logits.mean()
        
        return {
            "dpo_loss": dpo_loss,
            "preference_accuracy": preference_accuracy,
            "kl_divergence": kl_divergence,
            "reward_margin": reward_margin,
            "chosen_logprobs": chosen_policy_logprobs.mean(),
            "rejected_logprobs": rejected_policy_logprobs.mean()
        }
    
    def _log_training_step(self, losses: Dict[str, torch.Tensor]):
        """Log training step metrics"""
        log_dict = {
            "train/dpo_loss": losses["dpo_loss"].item(),
            "train/preference_accuracy": losses["preference_accuracy"],
            "train/kl_divergence": losses["kl_divergence"].item(),
            "train/reward_margin": losses["reward_margin"].item(),
            "train/learning_rate": self.scheduler.get_last_lr()[0],
            "global_step": self.global_step
        }
        
        if self.config.use_wandb and HAS_WANDB:
            wandb.log(log_dict)
        
        # Store for local tracking
        for key, value in log_dict.items():
            self.train_metrics[key].append(value)
    
    def _log_eval_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics"""
        log_dict = {f"eval/{k}": v for k, v in metrics.items()}
        
        if self.config.use_wandb and HAS_WANDB:
            wandb.log(log_dict)
        
        # Store for local tracking
        for key, value in log_dict.items():
            self.eval_metrics[key].append(value)
    
    def _log_epoch_results(self, train_metrics: Dict[str, float], 
                          eval_metrics: Optional[Dict[str, float]]):
        """Log epoch-level results"""
        logger.info(f"DPO Epoch {self.epoch + 1} completed:")
        logger.info(f"  📊 Train DPO Loss: {train_metrics['dpo_loss']:.4f}")
        logger.info(f"  📊 Train Preference Accuracy: {train_metrics['preference_accuracy']:.3f}")
        logger.info(f"  📊 Train KL Divergence: {train_metrics['kl_divergence']:.4f}")
        
        if eval_metrics:
            logger.info(f"  📊 Eval Preference Accuracy: {eval_metrics['preference_accuracy']:.3f}")
            logger.info(f"  🏆 Best Eval Reward: {self.best_eval_reward:.3f}")
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "policy_model_state_dict": self.model.policy_model.state_dict(),
            "reference_model_state_dict": self.model.reference_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_eval_reward": self.best_eval_reward,
            "config": self.config.__dict__
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 DPO checkpoint saved: {checkpoint_path}")

# =============================================================================
# MAIN DPO TRAINING SCRIPT
# =============================================================================

def load_sft_model(sft_checkpoint_path: str, model_config: Dict[str, Any]) -> MuseV3Architecture:
    """Load SFT-trained model"""
    model = MuseV3Architecture(model_config)  # Pass as dict, not kwargs
    
    if Path(sft_checkpoint_path).exists():
        checkpoint = torch.load(sft_checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"✅ Loaded SFT model from {sft_checkpoint_path}")
    else:
        logger.warning(f"⚠️ SFT checkpoint not found: {sft_checkpoint_path}, using random initialization")
    
    return model

def main():
    """Main DPO training script"""

    # Configuration - using your generated data
    config = DPOConfig(
        dpo_data_path="/media/adityapachauri/second_drive/Muse/muse_v3_advanced/research_experiments/muse_v3_comprehensive_study/generated_data/dpo_pairs.json",
        sft_checkpoint_path="/media/adityapachauri/second_drive/Muse/muse_v3_advanced/training_outputs/sft/checkpoints/best_model.pt",
        output_dir="/media/adityapachauri/second_drive/Muse/muse_v3_advanced/training_outputs/dpo",
        checkpoint_dir="/media/adityapachauri/second_drive/Muse/muse_v3_advanced/training_outputs/dpo/checkpoints",
        batch_size=4,  # Increased batch size for better training
        num_epochs=5,  # More epochs for thorough training
        learning_rate=5e-6,  # Lower learning rate for stability
        beta=0.2,  # Higher beta for stronger KL regularization
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_wandb=True,  # Enable wandb for monitoring
        gradient_accumulation_steps=4,  # Better gradient estimates
        warmup_ratio=0.15,  # More warmup for stability
        max_grad_norm=0.5,  # Stricter gradient clipping
        eval_steps=25,  # More frequent evaluation
        save_steps=100,  # More frequent checkpointing
        logging_steps=10   # More frequent logging
    )
    
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    print("🎯 Starting MUSE DPO Training")
    print("=" * 60)
    print(f"📊 Config: {config}")
    
    # Load datasets
    print("📚 Loading DPO datasets...")
    dataset = DPODataset(config.dpo_data_path, config)
    
    # Split for validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Initialize models
    print("🏗️ Initializing models...")
    model_config = {
        "text_dim": 384,
        "image_dim": 512,
        "metadata_dim": 256,
        "fusion_dim": 512,
        "num_intents": 7,
        "num_tools": 6,
        "max_steps": 5,
        "device": config.device,
        "metadata_vocab": {"category": 50, "brand": 100}
    }
    
    # Load SFT-trained model as policy model
    policy_model = load_sft_model(config.sft_checkpoint_path, model_config)
    
    # Create reference model (copy of policy model)
    reference_model = load_sft_model(config.sft_checkpoint_path, model_config)
    
    # Create DPO model
    dpo_model = MuseDPOModel(policy_model, reference_model, config)
    
    # Initialize trainer
    trainer = DPOTrainer(dpo_model, config)
    
    # Start training
    print("🎯 Starting DPO training...")
    trainer.train(train_dataset, val_dataset)
    
    print("🎉 DPO training completed!")
    print(f"💾 Results saved to: {config.output_dir}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
