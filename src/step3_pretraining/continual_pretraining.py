"""
ChipNeMo DAPT - Continual Pretraining with Proper Embedding Initialization
Optimized for small datasets to prevent gibberish outputs
"""

import yaml
import json
import torch
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)

import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.logger import setup_logger
    logger = setup_logger("chipnemo_dapt")
except:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("chipnemo_dapt")

warnings.filterwarnings('ignore')


class ChipNeMoCallback(TrainerCallback):
    """Monitor training progress with gibberish detection"""
    
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')
        self.initial_loss = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get('loss')
            eval_loss = logs.get('eval_loss')
            
            if loss is not None:
                # Record initial loss
                if self.initial_loss is None:
                    self.initial_loss = loss
                    logger.info(f"  Initial loss: {loss:.4f}")
                
                # Detect loss spikes (potential gibberish)
                if loss > 10.0:
                    logger.warning(f"  LOSS SPIKE at step {step}: {loss:.4f} - Model may produce gibberish!")
                
                # Detect loss explosion
                if self.initial_loss and loss > self.initial_loss * 3:
                    logger.error(f"  LOSS EXPLOSION at step {step}: {loss:.4f} (initial: {self.initial_loss:.4f})")
                    logger.error("   Training may be unstable - consider reducing learning rate")
                
                if loss < self.best_loss:
                    self.best_loss = loss
                
                if step % 50 == 0:
                    logger.info(f"  Step {step}: loss={loss:.4f}")
            
            if eval_loss is not None:
                logger.info(f"  Step {step}: eval_loss={eval_loss:.4f}")
    
    def on_save(self, args, state, control, **kwargs):
        logger.info(f"  Checkpoint saved at step {state.global_step} (best_loss={self.best_loss:.4f})")


class ChipNeMoDAPT:
    """ChipNeMo DAPT with proper embedding initialization"""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.data_config = self.config['data']
        self.lora_config = self.config.get('lora', {})
        self.early_stopping_config = self.config.get('early_stopping', {})
        
        self.device = self._setup_device()
        self._print_header()
    
    def _print_header(self):
        logger.info("=" * 70)
        logger.info("ChipNeMo DAPT - With Proper Embedding Initialization")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.model_config['base_model']}")
        logger.info("=" * 70)
    
    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"   Memory: {memory_gb:.2f}GB")
        else:
            device = 'cpu'
            logger.warning("  Using CPU")
        return device
    
    def load_initialized_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model from initialized_model directory (with proper embeddings)
        """
        logger.info("\n" + "=" * 70)
        logger.info("Loading Initialized Model & Tokenizer")
        logger.info("=" * 70)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("  GPU cache cleared")
        
        # CRITICAL: Load from initialized_model (not adapted_model)
        model_path = Path("models/initialized_model")
        if not model_path.exists():
            logger.error(f"  Initialized model not found: {model_path}")
            logger.error("  Run initialize_embeddings.py first!")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"  Loading model from: {model_path}")
        
        # Load tokenizer from same directory
        logger.info(f"  Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"  Tokenizer loaded: {len(tokenizer):,} tokens")
        
        # 4-bit quantization config
        quantization_config = self.model_config.get('quantization', {})
        bnb_config = None
        
        if quantization_config.get('load_in_4bit', False):
            logger.info("\n  Configuring 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True)
            )
            logger.info("   Type: 4-bit NormalFloat")
            logger.info("   Compute: bfloat16")
            logger.info("   Expected: ~8-10GB")
        
        # Load model
        attn_implementation = self.model_config.get('attn_implementation', 'sdpa')
        
        logger.info(f"\n  Loading model with proper embeddings...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            use_cache=False,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        logger.info("  Model loaded with initialized embeddings")
        
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"   GPU Memory: {allocated:.2f}GB")
        
        # Verify vocab size
        model_vocab_size = model.get_input_embeddings().num_embeddings
        tokenizer_vocab_size = len(tokenizer)
        
        if model_vocab_size != tokenizer_vocab_size:
            raise ValueError(
                f"   Vocab size mismatch!\n"
                f"   Model: {model_vocab_size:,}\n"
                f"   Tokenizer: {tokenizer_vocab_size:,}\n"
                f"   This will cause gibberish - check embedding initialization!"
            )
        
        logger.info(f"  Vocabulary verified: {tokenizer_vocab_size:,} tokens")
        
        # Verify embedding initialization (sample check)
        self._verify_embedding_initialization(model, tokenizer)
        
        # Apply LoRA
        if self.lora_config.get('enable', False):
            logger.info("\n  Applying LoRA...")
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            
            model = prepare_model_for_kbit_training(model)
            
            peft_config = LoraConfig(
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['lora_alpha'],
                target_modules=self.lora_config['target_modules'],
                lora_dropout=self.lora_config['lora_dropout'],
                bias=self.lora_config['bias'],
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            if self.device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"   GPU after LoRA: {allocated:.2f}GB")
        
        # Stats
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"\n  Model Statistics:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"\n  GPU Memory Status:")
            logger.info(f"   Used: {allocated:.2f}GB / {total:.2f}GB")
            logger.info(f"   Available: {total - reserved:.2f}GB")
        
        logger.info("=" * 70)
        
        return model, tokenizer
    
    def _verify_embedding_initialization(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Verify that new token embeddings are properly initialized
        """
        logger.info("\n  Verifying Embedding Initialization...")
        
        # Check if we have the initialization report
        report_path = Path("models/initialized_model/embedding_initialization_report.json")
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            base_vocab = report.get('base_vocab_size', 0)
            adapted_vocab = report.get('adapted_vocab_size', 0)
            num_new = report.get('num_new_tokens', 0)
            
            logger.info(f"   Base vocab: {base_vocab:,}")
            logger.info(f"   Adapted vocab: {adapted_vocab:,}")
            logger.info(f"   New tokens: {num_new:,}")
            
            # Quick sanity check on a few new tokens
            if num_new > 0:
                input_embeddings = model.get_input_embeddings().weight.data
                
                # Check last 3 new tokens
                sample_ids = list(range(base_vocab, min(base_vocab + 3, adapted_vocab)))
                all_good = True
                
                for token_id in sample_ids:
                    norm = torch.norm(input_embeddings[token_id]).item()
                    if norm < 1e-6:
                        all_good = False
                        logger.warning(f"     Token {token_id} has zero embedding!")
                
                if all_good:
                    logger.info("     Sample embeddings look good (non-zero)")
                else:
                    logger.error("     Some embeddings are zero - model will produce gibberish!")
                    raise ValueError("Embedding initialization failed - run initialize_embeddings.py")
        else:
            logger.warning("     No initialization report found - cannot verify")
    
    def prepare_dataset(self, tokenizer: AutoTokenizer) -> DatasetDict:
        """Prepare and tokenize dataset"""
        logger.info("\n" + "=" * 70)
        logger.info("Preparing Dataset")
        logger.info("=" * 70)
        
        curated_file = Path(self.data_config['curated_data_path'])
        if not curated_file.exists():
            raise FileNotFoundError(f"  Data not found: {curated_file}")
        
        logger.info(f"  Loading: {curated_file}")
        
        # Check if split already exists
        processed_dir = Path('data/processed')
        train_file = processed_dir / 'train.jsonl'
        val_file = processed_dir / 'val.jsonl'
        
        if not (train_file.exists() and val_file.exists()):
            logger.info("  Creating train/val split...")
            self._create_split(curated_file, processed_dir)
        
        # Load dataset
        dataset = load_dataset('json', data_files={
            'train': str(train_file),
            'validation': str(val_file)
        })
        
        logger.info(f"   Dataset loaded:")
        logger.info(f"   Train samples: {len(dataset['train']):,}")
        logger.info(f"   Validation samples: {len(dataset['validation']):,}")
        
        # Small dataset warning
        if len(dataset['train']) < 100:
            logger.warning(f"\n   SMALL DATASET WARNING: Only {len(dataset['train'])} training samples!")
            logger.warning("   Recommendations:")
            logger.warning("   - Use higher learning rate (5e-4 to 1e-3)")
            logger.warning("   - Increase epochs (10-20)")
            logger.warning("   - Enable eval_on_start=true")
            logger.warning("   - Use load_best_model_at_end=true")
            logger.warning("   - Monitor for overfitting closely\n")
        
        # Tokenization
        context_length = self.model_config.get('context_length', 2048)
        
        def tokenize_function(examples):
            outputs = tokenizer(
                examples['text'],
                truncation=True,
                max_length=context_length,
                padding='max_length',
                return_tensors=None
            )
            outputs['labels'] = outputs['input_ids'].copy()
            return outputs
        
        logger.info(f"\n  Tokenizing (context_length={context_length})...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing"
        )
        
        logger.info(f"  Tokenization complete")
        total_tokens = len(tokenized_dataset['train']) * context_length
        logger.info(f"   Total training tokens: ~{total_tokens/1e6:.2f}M")
        
        return tokenized_dataset
    
    def _create_split(self, curated_file: Path, output_dir: Path):
        """Create train/validation split"""
        import random
        
        all_data = []
        with open(curated_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'text' in data and len(data['text'].strip()) > 0:
                        all_data.append(data)
                except:
                    continue
        
        logger.info(f"   Loaded {len(all_data):,} samples")
        
        random.seed(42)
        random.shuffle(all_data)
        
        split_ratio = self.data_config.get('train_val_split', 0.9)
        split_idx = int(len(all_data) * split_ratio)
        
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'train.jsonl', 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps({'text': item['text']}) + '\n')
        
        with open(output_dir / 'val.jsonl', 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps({'text': item['text']}) + '\n')
        
        logger.info(f"     Split created: {len(train_data):,} train, {len(val_data):,} validation")
    
    def train(self):
        """Execute continual pretraining"""
        logger.info("\n" + "=" * 70)
        logger.info("  Starting Continual Pretraining")
        logger.info("=" * 70)
        
        # Load model and tokenizer
        model, tokenizer = self.load_initialized_model_and_tokenizer()
        
        # Prepare dataset
        tokenized_dataset = self.prepare_dataset(tokenizer)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Training arguments
        logger.info("\n" + "=" * 70)
        logger.info("Training Configuration")
        logger.info("=" * 70)
        
        training_args = TrainingArguments(
            output_dir=self.training_config['output_dir'],
            learning_rate=self.training_config['learning_rate'],
            lr_scheduler_type=self.training_config['lr_scheduler_type'],
            warmup_ratio=self.training_config['warmup_ratio'],
            weight_decay=self.training_config['weight_decay'],
            adam_beta1=self.training_config['adam_beta1'],
            adam_beta2=self.training_config['adam_beta2'],
            adam_epsilon=self.training_config['adam_epsilon'],
            max_grad_norm=self.training_config['max_grad_norm'],
            optim=self.training_config['optim'],
            num_train_epochs=self.training_config['num_train_epochs'],
            max_steps=self.training_config.get('max_steps', -1),
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            bf16=self.training_config.get('bf16', True),
            fp16=self.training_config.get('fp16', False),
            tf32=self.training_config.get('tf32', True),
            eval_strategy=self.training_config['eval_strategy'],
            eval_steps=self.training_config['eval_steps'],
            eval_accumulation_steps=self.training_config.get('eval_accumulation_steps', 4),
            eval_on_start=self.training_config.get('eval_on_start', True),
            save_strategy=self.training_config['save_strategy'],
            save_steps=self.training_config['save_steps'],
            save_total_limit=self.training_config['save_total_limit'],
            save_only_model=self.training_config.get('save_only_model', False),
            load_best_model_at_end=self.training_config.get('load_best_model_at_end', True),
            metric_for_best_model=self.training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=self.training_config.get('greater_is_better', False),
            logging_dir=self.training_config.get('logging_dir', f"{self.training_config['output_dir']}/logs"),
            logging_steps=self.training_config['logging_steps'],
            logging_first_step=self.training_config.get('logging_first_step', True),
            logging_strategy=self.training_config.get('logging_strategy', 'steps'),
            report_to=self.training_config.get('report_to', ['tensorboard']),
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', True),
            gradient_checkpointing_kwargs=self.training_config.get('gradient_checkpointing_kwargs', {'use_reentrant': False}),
            dataloader_num_workers=self.training_config.get('dataloader_num_workers', 2),
            dataloader_pin_memory=self.training_config.get('dataloader_pin_memory', False),
            dataloader_prefetch_factor=self.training_config.get('dataloader_prefetch_factor', 2),
            dataloader_drop_last=self.training_config.get('dataloader_drop_last', False),
            group_by_length=self.training_config.get('group_by_length', False),
            seed=self.training_config.get('seed', 42),
            no_cuda=(self.device == 'cpu'),
            remove_unused_columns=False,
            prediction_loss_only=self.training_config.get('prediction_loss_only', False),
            label_names=self.training_config.get('label_names', ['labels']),
        )
        
        # Calculate training stats
        effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        steps_per_epoch = len(tokenized_dataset['train']) // effective_batch
        total_steps = steps_per_epoch * training_args.num_train_epochs if training_args.max_steps == -1 else training_args.max_steps
        
        logger.info(f"   Learning rate: {training_args.learning_rate:.2e}")
        logger.info(f"   Per-device batch: {training_args.per_device_train_batch_size}")
        logger.info(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
        logger.info(f"   Effective batch size: {effective_batch}")
        logger.info(f"   Epochs: {training_args.num_train_epochs}")
        logger.info(f"   Steps per epoch: ~{steps_per_epoch}")
        logger.info(f"   Total steps: ~{total_steps}")
        logger.info(f"   Warmup ratio: {training_args.warmup_ratio}")
        logger.info(f"   Weight decay: {training_args.weight_decay}")
        logger.info(f"   Max grad norm: {training_args.max_grad_norm}")
        
        # Callbacks
        callbacks = [ChipNeMoCallback()]
        
        if self.early_stopping_config.get('enable', False):
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_config.get('patience', 10),
                early_stopping_threshold=self.early_stopping_config.get('threshold', 0.01)
            )
            callbacks.append(early_stopping)
            logger.info(f"   Early stopping: patience={self.early_stopping_config.get('patience', 10)}")
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("  Starting Training...")
        logger.info("=" * 70)
        logger.info("  Monitor loss carefully - spikes may indicate gibberish generation")
        logger.info("=" * 70 + "\n")
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        final_model_path = Path(self.training_config['output_dir']) / "chipnemo_final"
        final_model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n  Saving final model to: {final_model_path}")
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        # Save metrics
        metrics = train_result.metrics
        with open(final_model_path / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        with open(final_model_path / "training_config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("  Training Complete!")
        logger.info("=" * 70)
        logger.info(f"   Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"   Training time: {metrics.get('train_runtime', 0)/3600:.2f} hours")
        logger.info(f"   Model saved to: {final_model_path}")
        logger.info("=" * 70)
        logger.info("\n  Next steps:")
        logger.info("   1. Test model outputs for quality")
        logger.info("   2. Compare with base model performance")
        logger.info("   3. Run your chatbot evaluation")
        logger.info("=" * 70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ChipNeMo Continual Pretraining")
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training config file')
    args = parser.parse_args()
    
    trainer = ChipNeMoDAPT(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
