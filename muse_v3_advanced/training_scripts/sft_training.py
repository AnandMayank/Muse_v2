#!/usr/bin/env python3
"""
MUSE v3 Supervised Fine-Tuning (SFT) Training Script
===================================================

Full-scale SFT training implementation following Toolformer methodology:
1. Real Toolformer-style data loading and processing
2. Actual gradient-based training with proper loss functions
3. Tool call position and argument prediction
4. Comprehensive evaluation and checkpointing

Based on Toolformer (arXiv:2302.04761) self-supervised learning approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import time
from collections import defaultdict
import re

# Handle optional imports
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, **kwargs):
        return LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_warmup_steps)

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

# Import MUSE architecture
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture import MuseV3Architecture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SFTConfig:
    """Complete SFT training configuration"""
    
    # Data paths
    toolformer_data_path: str
    output_dir: str
    checkpoint_dir: str
    
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Model parameters
    max_sequence_length: int = 512
    tool_vocab_size: int = 6
    
    # Loss weights
    tool_prediction_weight: float = 1.0
    position_prediction_weight: float = 0.8
    argument_prediction_weight: float = 0.6
    fluency_weight: float = 0.4
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    
    # System
    device: str = "cpu"
    num_workers: int = 4
    seed: int = 42
    
    # Wandb logging
    use_wandb: bool = True
    project_name: str = "muse-sft-training"
    experiment_name: str = "toolformer-sft"

# =============================================================================
# DATASET IMPLEMENTATION
# =============================================================================

class ToolformerSFTDataset(Dataset):
    """
    Advanced Toolformer SFT dataset with proper tokenization and labeling
    
    Features:
    - Tool call position prediction
    - Tool type classification
    - Tool argument generation
    - Text fluency preservation
    """
    
    def __init__(self, data_path: str, config: SFTConfig, split: str = "train"):
        self.config = config
        self.split = split
        
        # Load Toolformer data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # Filter valid examples (with tool calls)
        self.data = [item for item in self.raw_data if item.get("tool_calls") and len(item["tool_calls"]) > 0]
        
        # Tool vocabulary
        self.tool_to_id = {
            "search": 0, "recommend": 1, "compare": 2,
            "filter": 3, "translate": 4, "visual_search": 5
        }
        self.id_to_tool = {v: k for k, v in self.tool_to_id.items()}
        
        # Build vocabulary from data
        self._build_vocabulary()
        
        logger.info(f"📚 Loaded {len(self.data)} valid Toolformer examples for {split}")
        
    def _build_vocabulary(self):
        """Build vocabulary from training data"""
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<TOOL>": 2, "<ARG>": 3, "<END>": 4}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
        word_freq = defaultdict(int)
        
        # Count word frequencies
        for item in self.data:
            words = self._tokenize_text(item["original_text"])
            for word in words:
                word_freq[word] += 1
        
        # Add frequent words to vocabulary
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if len(self.word_to_id) >= 10000:  # Vocabulary limit
                break
            if freq >= 2:  # Minimum frequency
                self.word_to_id[word] = len(self.word_to_id)
                self.id_to_word[len(self.id_to_word)] = word
        
        logger.info(f"Built vocabulary with {len(self.word_to_id)} tokens")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for text"""
        # Basic tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process original and augmented text
        original_tokens = self._tokenize_text(item["original_text"])
        augmented_tokens = self._tokenize_text(item["augmented_text"])
        
        # Create training example
        example = self._create_training_example(item, original_tokens, augmented_tokens)
        
        return example
    
    def _create_training_example(self, item: Dict[str, Any], 
                               original_tokens: List[str],
                               augmented_tokens: List[str]) -> Dict[str, torch.Tensor]:
        """Create comprehensive training example with multiple prediction targets"""
        
        # Convert tokens to IDs
        input_ids = self._tokens_to_ids(augmented_tokens)
        
        # Create tool prediction labels
        tool_labels = self._create_tool_labels(item["tool_calls"])
        
        # Create position labels for tool insertion points
        position_labels = self._create_position_labels(item, augmented_tokens)
        
        # Create argument prediction targets
        argument_labels = self._create_argument_labels(item["tool_calls"])
        
        # Pad sequences
        input_ids = self._pad_sequence(input_ids, self.config.max_sequence_length)
        position_labels = self._pad_sequence(position_labels, self.config.max_sequence_length)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "tool_labels": torch.tensor(tool_labels, dtype=torch.long),
            "position_labels": torch.tensor(position_labels, dtype=torch.float),
            "argument_labels": argument_labels,
            "attention_mask": torch.tensor([1 if x != 0 else 0 for x in input_ids], dtype=torch.long),
            "original_text": item["original_text"],
            "quality_score": item.get("quality_score", 0.0)
        }
    
    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to vocabulary IDs"""
        return [self.word_to_id.get(token, self.word_to_id["<UNK>"]) for token in tokens]
    
    def _create_tool_labels(self, tool_calls: List[Dict[str, Any]]) -> List[int]:
        """Create tool type classification labels"""
        labels = []
        for tool_call in tool_calls:
            tool_name = tool_call["tool_name"]
            tool_id = self.tool_to_id.get(tool_name, 0)
            labels.append(tool_id)
        
        # Pad to consistent length (max 3 tool calls)
        while len(labels) < 3:
            labels.append(-1)  # -1 for padding
        
        return labels[:3]  # Take max 3 tool calls
    
    def _create_position_labels(self, item: Dict[str, Any], 
                              augmented_tokens: List[str]) -> List[float]:
        """Create position prediction labels for tool insertion"""
        labels = [0.0] * len(augmented_tokens)
        
        # Mark tool call positions
        for tool_call in item["tool_calls"]:
            position = tool_call.get("position", 0)
            # Approximate token position
            token_position = min(position // 5, len(labels) - 1)  # Rough estimate
            labels[token_position] = 1.0
        
        return labels
    
    def _create_argument_labels(self, tool_calls: List[Dict[str, Any]]) -> torch.Tensor:
        """Create argument prediction labels"""
        # Simplified argument representation
        arg_features = []
        
        for tool_call in tool_calls:
            args = tool_call.get("arguments", {})
            
            # Convert arguments to numerical features
            features = [
                len(str(args.get("query", ""))),  # Query length
                args.get("max_results", 10) / 100.0,  # Normalized max results
                float(args.get("similarity_threshold", 0.5)),  # Similarity threshold
                float("num_items" in args),  # Has num_items flag
                float("user_profile" in args),  # Has user_profile flag
            ]
            
            arg_features.extend(features)
        
        # Pad to consistent size (3 tool calls * 5 features)
        while len(arg_features) < 15:
            arg_features.append(0.0)
        
        return torch.tensor(arg_features[:15], dtype=torch.float)
    
    def _pad_sequence(self, sequence: List, max_length: int, pad_value: int = 0):
        """Pad sequence to max length"""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        
        return sequence + [pad_value] * (max_length - len(sequence))

# =============================================================================
# MODEL WRAPPER FOR SFT
# =============================================================================

class MuseSFTModel(nn.Module):
    """
    MUSE model wrapper for SFT training with multiple prediction heads
    """
    
    def __init__(self, base_model: MuseV3Architecture, config: SFTConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Additional prediction heads for SFT
        hidden_dim = base_model.config.get("fusion_dim", 512)
        
        # Tool call position prediction head
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Tool argument prediction head
        self.argument_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 15)  # 3 tools * 5 features
        )
        
        # Text fluency head
        self.fluency_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Prepare input for base model
        model_input = {
            "text_input": batch["input_ids"],
            "batch_size": batch["input_ids"].size(0),
            "metadata_categorical": {
                "category": torch.zeros(batch["input_ids"].size(0), dtype=torch.long, device=batch["input_ids"].device),
                "brand": torch.zeros(batch["input_ids"].size(0), dtype=torch.long, device=batch["input_ids"].device)
            }
        }
        
        # Get base model outputs
        base_outputs = self.base_model(model_input)
        
        # Get fusion features for additional predictions
        fusion_features = base_outputs.get("fusion_features", torch.zeros(batch["input_ids"].size(0), self.config.max_sequence_length, self.base_model.config.get("fusion_dim", 512), device=batch["input_ids"].device))
        
        # Position predictions (per token)
        position_logits = self.position_predictor(fusion_features).squeeze(-1)
        
        # Argument predictions (per sequence)
        pooled_features = fusion_features.mean(dim=1)
        argument_logits = self.argument_predictor(pooled_features)
        
        # Fluency predictions
        fluency_logits = self.fluency_predictor(pooled_features).squeeze(-1)
        
        return {
            "intent_logits": base_outputs["intent_logits"],
            "position_logits": position_logits,
            "argument_logits": argument_logits,
            "fluency_logits": fluency_logits,
            "fusion_features": fusion_features
        }

# =============================================================================
# TRAINER IMPLEMENTATION
# =============================================================================

class SFTTrainer:
    """
    Comprehensive SFT trainer with proper loss functions and evaluation
    """
    
    def __init__(self, model: MuseSFTModel, config: SFTConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss functions
        self.tool_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.position_loss = nn.BCELoss()
        self.argument_loss = nn.MSELoss()
        self.fluency_loss = nn.BCELoss()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
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
        
        logger.info("🏋️ SFT Trainer initialized")
    
    def train(self, train_dataset: ToolformerSFTDataset, 
              eval_dataset: Optional[ToolformerSFTDataset] = None):
        """Main training loop"""
        
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
        
        logger.info(f"🚀 Starting SFT training for {self.config.num_epochs} epochs")
        logger.info(f"📊 Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Evaluation phase
            if eval_loader:
                eval_loss = self._eval_epoch(eval_loader)
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint("best_model.pt")
            
            # Log epoch results
            self._log_epoch_results(train_loss, eval_loss if eval_loader else None)
            
            # Save regular checkpoint
            self._save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        logger.info("✅ SFT training completed!")
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate losses
            losses = self._calculate_losses(outputs, batch)
            total_loss = sum(losses.values())
            
            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                total_loss = total_loss / self.config.gradient_accumulation_steps
            
            total_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_training_step(losses, total_loss)
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return epoch_loss / num_batches
    
    def _eval_epoch(self, eval_loader: DataLoader) -> float:
        """Evaluate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate losses
                losses = self._calculate_losses(outputs, batch)
                total_loss = sum(losses.values())
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                all_predictions.append(outputs["intent_logits"].cpu())
                all_labels.append(batch["tool_labels"].cpu())
        
        # Calculate evaluation metrics
        eval_metrics = self._calculate_eval_metrics(all_predictions, all_labels)
        self._log_eval_metrics(eval_metrics)
        
        return epoch_loss / num_batches
    
    def _calculate_losses(self, outputs: Dict[str, torch.Tensor], 
                         batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate all training losses"""
        losses = {}
        
        # Tool prediction loss
        tool_logits = outputs["intent_logits"]
        tool_labels = batch["tool_labels"][:, 0]  # Use first tool label
        valid_mask = tool_labels >= 0
        
        if valid_mask.sum() > 0:
            losses["tool_loss"] = self.tool_loss(
                tool_logits[valid_mask], 
                tool_labels[valid_mask]
            ) * self.config.tool_prediction_weight
        else:
            losses["tool_loss"] = torch.tensor(0.0, device=self.device)
        
        # Position prediction loss
        if "position_logits" in outputs:
            position_logits = outputs["position_logits"]
            position_labels = batch["position_labels"]
            
            losses["position_loss"] = self.position_loss(
                position_logits, position_labels
            ) * self.config.position_prediction_weight
        
        # Argument prediction loss
        if "argument_logits" in outputs:
            argument_logits = outputs["argument_logits"]
            argument_labels = batch["argument_labels"]
            
            losses["argument_loss"] = self.argument_loss(
                argument_logits, argument_labels
            ) * self.config.argument_prediction_weight
        
        # Fluency loss (predict high fluency for quality examples)
        if "fluency_logits" in outputs:
            fluency_logits = outputs["fluency_logits"]
            fluency_targets = torch.tensor([
                min(1.0, score * 2) for score in batch["quality_score"]
            ], dtype=torch.float32, device=self.device)
            
            losses["fluency_loss"] = self.fluency_loss(
                fluency_logits, fluency_targets
            ) * self.config.fluency_weight
        
        return losses
    
    def _calculate_eval_metrics(self, predictions: List[torch.Tensor], 
                              labels: List[torch.Tensor]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        # Concatenate all predictions and labels
        all_preds = torch.cat(predictions, dim=0)
        all_labels = torch.cat(labels, dim=0)
        
        # Tool prediction accuracy
        pred_tools = all_preds.argmax(dim=-1)
        actual_tools = all_labels[:, 0]
        valid_mask = actual_tools >= 0
        
        tool_accuracy = (pred_tools[valid_mask] == actual_tools[valid_mask]).float().mean().item()
        
        # Per-tool accuracy
        tool_accuracies = {}
        for tool_id in range(6):
            tool_mask = (actual_tools == tool_id) & valid_mask
            if tool_mask.sum() > 0:
                tool_acc = (pred_tools[tool_mask] == tool_id).float().mean().item()
                tool_accuracies[f"tool_{tool_id}_accuracy"] = tool_acc
        
        metrics = {
            "tool_accuracy": tool_accuracy,
            **tool_accuracies
        }
        
        return metrics
    
    def _log_training_step(self, losses: Dict[str, torch.Tensor], total_loss: torch.Tensor):
        """Log training step metrics"""
        log_dict = {
            "train/total_loss": total_loss.item(),
            "train/learning_rate": self.scheduler.get_last_lr()[0],
            "global_step": self.global_step
        }
        
        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.item()
        
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
    
    def _log_epoch_results(self, train_loss: float, eval_loss: Optional[float]):
        """Log epoch-level results"""
        logger.info(f"Epoch {self.epoch + 1} completed:")
        logger.info(f"  📊 Train Loss: {train_loss:.4f}")
        
        if eval_loss is not None:
            logger.info(f"  📊 Eval Loss: {eval_loss:.4f}")
            logger.info(f"  🏆 Best Eval Loss: {self.best_eval_loss:.4f}")
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.__dict__
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main SFT training script"""
    
    # Configuration
    config = SFTConfig(
        toolformer_data_path="/media/adityapachauri/second_drive/Muse/muse_v3_advanced/research_experiments/muse_v3_comprehensive_study/generated_data/toolformer_augmented.json",
        output_dir="/media/adityapachauri/second_drive/Muse/muse_v3_advanced/training_outputs/sft",
        checkpoint_dir="/media/adityapachauri/second_drive/Muse/muse_v3_advanced/training_outputs/sft/checkpoints",
        batch_size=4,  # Adjusted for CPU
        num_epochs=20,  # Increase epochs for longer training
        learning_rate=5e-5,
        device="cpu",
        use_wandb=True  # Enable wandb logging
    )
    
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting MUSE SFT Training")
    print("=" * 60)
    print(f"📊 Config: {config}")
    
    # Load datasets
    print("📚 Loading datasets...")
    train_dataset = ToolformerSFTDataset(config.toolformer_data_path, config, "train")
    
    # Split for validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Initialize base model
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
    
    base_model = MuseV3Architecture(model_config)
    sft_model = MuseSFTModel(base_model, config)
    
    # Initialize trainer
    trainer = SFTTrainer(sft_model, config)
    
    # Start training
    print("🏋️ Starting training...")
    trainer.train(train_dataset, val_dataset)
    
    print("🎉 SFT training completed!")
    print(f"💾 Results saved to: {config.output_dir}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
