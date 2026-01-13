"""
Model Evaluation & Comparison Script
Compares base, initialized, and pretrained models on validation set
Generates loss/perplexity plots and quality metrics
FIXED: Handles HuggingFace model IDs properly
"""

import torch
import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import warnings
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.logger import setup_logger
    logger = setup_logger("model_evaluator")
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("model_evaluator")

warnings.filterwarnings('ignore')


def is_huggingface_model_id(path: str) -> bool:
    """Check if path is a HuggingFace model ID vs local path"""
    # HuggingFace IDs have format: org/model-name
    # Local paths would be actual directories
    return '/' in path and not Path(path).exists()


class ModelEvaluator:
    """
    Evaluate and compare multiple model variants
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.model_config = self.config['model']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info("=" * 70)
        logger.info("  Model Evaluation & Comparison")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        if self.device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info("=" * 70 + "\n")
    
    def load_validation_data(self, max_samples: int = 100) -> List[str]:
        """Load validation texts"""
        logger.info("  Loading validation data...")
        
        val_file = Path('data/processed/val.jsonl')
        if not val_file.exists():
            # Fallback to curated data
            val_file = Path(self.data_config['curated_data_path'])
        
        texts = []
        with open(val_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if text and len(text.strip()) > 50:
                        texts.append(text)
                except:
                    continue
        
        logger.info(f"  Loaded {len(texts)} validation samples\n")
        return texts
    
    def compute_loss_and_perplexity(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        batch_size: int = 4,
        max_length: int = 512
    ) -> Tuple[float, float, List[float]]:
        """
        Compute average loss and perplexity on validation texts
        Returns: (avg_loss, perplexity, per_sample_losses)
        """
        model.eval()
        all_losses = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # Forward pass
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    # Get per-sample losses
                    loss = outputs.loss
                    all_losses.append(loss.item())
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {i}: {e}")
                    continue
        
        if not all_losses:
            return float('inf'), float('inf'), []
        
        avg_loss = np.mean(all_losses)
        perplexity = np.exp(avg_loss)
        
        return avg_loss, perplexity, all_losses
    
    def evaluate_model(
        self,
        model_path: str,
        model_name: str,
        validation_texts: List[str],
        is_hf_model: bool = False
    ) -> Dict:
        """
        Evaluate a single model
        
        Args:
            model_path: Path to model (local directory or HuggingFace ID)
            model_name: Display name for the model
            validation_texts: List of validation text samples
            is_hf_model: True if model_path is a HuggingFace model ID
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 70)
        logger.info(f"Evaluating: {model_name}")
        logger.info("=" * 70)
        logger.info(f"Path: {model_path}")
        
        # Check if path exists (skip check for HuggingFace models)
        if not is_hf_model and not Path(model_path).exists():
            logger.error(f"  Model not found: {model_path}")
            return {
                'name': model_name,
                'path': model_path,
                'exists': False,
                'avg_loss': float('inf'),
                'perplexity': float('inf'),
                'per_sample_losses': []
            }
        
        try:
            # Determine load paths
            if is_hf_model:
                logger.info(f"   Loading from HuggingFace Hub: {model_path}")
                tokenizer_path = model_path
                model_load_path = model_path
            else:
                logger.info(f"   Loading from local path: {model_path}")
                tokenizer_path = model_path
                model_load_path = model_path
            
            # Load tokenizer
            logger.info("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"  Tokenizer loaded: {len(tokenizer):,} tokens")
            
            # Load model
            logger.info("  Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_load_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"  Model loaded successfully")
            
            if self.device == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"   GPU Memory: {allocated:.2f}GB")
            
            # Compute metrics
            logger.info("\n  Computing loss and perplexity...")
            avg_loss, perplexity, per_sample_losses = self.compute_loss_and_perplexity(
                model, tokenizer, validation_texts
            )
            
            logger.info(f"\n  Results:")
            logger.info(f"   Average Loss: {avg_loss:.4f}")
            logger.info(f"   Perplexity: {perplexity:.2f}")
            logger.info(f"   Vocab Size: {len(tokenizer):,}")
            
            # Memory cleanup
            del model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info("=" * 70 + "\n")
            
            return {
                'name': model_name,
                'path': model_path,
                'exists': True,
                'avg_loss': avg_loss,
                'perplexity': perplexity,
                'per_sample_losses': per_sample_losses,
                'vocab_size': len(tokenizer)
            }
            
        except Exception as e:
            logger.error(f"  Error evaluating {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'name': model_name,
                'path': model_path,
                'exists': True,
                'error': str(e),
                'avg_loss': float('inf'),
                'perplexity': float('inf'),
                'per_sample_losses': []
            }
    
    def compare_models(
        self,
        base_model_path: str = None,
        initialized_model_path: str = "models/initialized_model",
        pretrained_model_path: str = "models/checkpoints/chipnemo_final",
        max_val_samples: int = 100
    ) -> Dict:
        """
        Compare all three model variants
        
        Args:
            base_model_path: Path to base model (HuggingFace ID or local path)
            initialized_model_path: Path to initialized model (local)
            pretrained_model_path: Path to pretrained model (local)
            max_val_samples: Maximum validation samples to use
        
        Returns:
            Dictionary with results for all models
        """
        logger.info("\n" + "=" * 70)
        logger.info("  Starting Model Comparison")
        logger.info("=" * 70 + "\n")
        
        # Load validation data
        validation_texts = self.load_validation_data(max_samples=max_val_samples)
        
        if not validation_texts:
            logger.error("  No validation data found!")
            return {}
        
        # Use base model from config if not provided
        if base_model_path is None:
            base_model_path = self.model_config['base_model']
        
        # Check if base model is a HuggingFace ID
        is_base_hf_model = is_huggingface_model_id(base_model_path)
        
        if is_base_hf_model:
            logger.info(f"  Base model will be loaded from HuggingFace: {base_model_path}")
        else:
            logger.info(f"  Base model will be loaded from local path: {base_model_path}")
        
        # Evaluate each model
        results = {}
        
        # 1. Base Model (from HuggingFace or local)
        logger.info("\n" + "- " * 35)
        logger.info("  Model 1/3: Base Model (No Adaptation)")
        logger.info(" " * 35)
        results['base'] = self.evaluate_model(
            model_path=base_model_path,
            model_name="Base Model",
            validation_texts=validation_texts,
            is_hf_model=is_base_hf_model
        )
        
        # 2. Initialized Model (local)
        logger.info("\n" + "" * 35)
        logger.info("  Model 2/3: Initialized Model")
        logger.info("  (Adapted Tokenizer + Initialized Embeddings)")
        logger.info(" " * 35)
        results['initialized'] = self.evaluate_model(
            model_path=initialized_model_path,
            model_name="Initialized Model",
            validation_texts=validation_texts,
            is_hf_model=False
        )
        
        # 3. Pretrained Model (local)
        logger.info("\n" + " " * 35)
        logger.info("  Model 3/3: Continually Pretrained Model")
        logger.info(" " * 35)
        results['pretrained'] = self.evaluate_model(
            model_path=pretrained_model_path,
            model_name="Pretrained Model",
            validation_texts=validation_texts,
            is_hf_model=False
        )
        
        return results
    
    def plot_comparison(
        self,
        results: Dict,
        save_dir: str = "results/evaluation"
    ):
        """
        Generate comparison plots
        """
        logger.info("\n" + "=" * 70)
        logger.info("  Generating Comparison Plots")
        logger.info("=" * 70)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        models = []
        losses = []
        perplexities = []
        vocab_sizes = []
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
        
        for key, color in zip(['base', 'initialized', 'pretrained'], colors):
            if key in results and results[key]['exists']:
                models.append(results[key]['name'])
                losses.append(results[key]['avg_loss'])
                perplexities.append(results[key]['perplexity'])
                vocab_sizes.append(results[key].get('vocab_size', 0))
        
        if not models:
            logger.error("  No valid results to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison: Base vs Initialized vs Pretrained', 
                     fontsize=16, fontweight='bold')
        
        # 1. Average Loss Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, losses, color=colors[:len(models)], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Average Validation Loss', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, loss in zip(bars1, losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Perplexity Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, perplexities, color=colors[:len(models)], alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
        ax2.set_title('Perplexity (Lower is Better)', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, ppl in zip(bars2, perplexities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ppl:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Loss Distribution (Box Plot)
        ax3 = axes[1, 0]
        loss_distributions = []
        labels = []
        for key, result in results.items():
            if result['exists'] and result['per_sample_losses']:
                loss_distributions.append(result['per_sample_losses'])
                labels.append(result['name'])
        
        if loss_distributions:
            bp = ax3.boxplot(loss_distributions, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(labels)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax3.set_title('Loss Distribution per Sample', fontsize=13, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Vocabulary Size Comparison
        ax4 = axes[1, 1]
        bars4 = ax4.bar(models, vocab_sizes, color=colors[:len(models)], alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Vocabulary Size', fontsize=12, fontweight='bold')
        ax4.set_title('Model Vocabulary Size', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, vocab in zip(bars4, vocab_sizes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{vocab:,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = save_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Plot saved: {plot_path}")
        
        # Also save as PDF
        pdf_path = save_dir / "model_comparison.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        logger.info(f"  PDF saved: {pdf_path}")
        
        plt.close()
        
        # Create additional plot: Improvement percentage
        if len(models) >= 2:
            self._plot_improvement_metrics(results, save_dir)
    
    def _plot_improvement_metrics(self, results: Dict, save_dir: Path):
        """Plot improvement percentages"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        base_loss = results['base']['avg_loss']
        
        improvements = []
        labels = []
        colors_imp = []
        
        for key, name, color in [
            ('initialized', 'Initialized\nvs Base', '#e74c3c'),
            ('pretrained', 'Pretrained\nvs Base', '#2ecc71')
        ]:
            if key in results and results[key]['exists']:
                current_loss = results[key]['avg_loss']
                improvement = ((base_loss - current_loss) / base_loss) * 100
                improvements.append(improvement)
                labels.append(name)
                colors_imp.append(color)
        
        bars = ax.barh(labels, improvements, color=colors_imp, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Loss Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title('Loss Improvement Compared to Base Model', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            label_x = width + (1 if width > 0 else -1)
            ax.text(label_x, bar.get_y() + bar.get_height()/2.,
                   f'{imp:+.2f}%',
                   ha='left' if width > 0 else 'right',
                   va='center',
                   fontsize=11,
                   fontweight='bold')
        
        plt.tight_layout()
        
        improvement_path = save_dir / "improvement_metrics.png"
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Improvement plot saved: {improvement_path}")
        plt.close()
    
    def save_results(
        self,
        results: Dict,
        save_dir: str = "results/evaluation"
    ):
        """Save evaluation results to JSON"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove per-sample losses from JSON (too large)
        results_to_save = {}
        for key, value in results.items():
            results_to_save[key] = {k: v for k, v in value.items() 
                                   if k != 'per_sample_losses'}
        
        results_path = save_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"  Results saved: {results_path}")
        
        # Create summary report
        self._create_summary_report(results, save_dir)
    
    def _create_summary_report(self, results: Dict, save_dir: Path):
        """Create human-readable summary report"""
        report_path = save_dir / "evaluation_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for key in ['base', 'initialized', 'pretrained']:
                if key in results and results[key]['exists']:
                    result = results[key]
                    f.write(f"{result['name']}\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"  Path: {result['path']}\n")
                    f.write(f"  Vocabulary Size: {result.get('vocab_size', 'N/A'):,}\n")
                    f.write(f"  Average Loss: {result['avg_loss']:.4f}\n")
                    f.write(f"  Perplexity: {result['perplexity']:.2f}\n")
                    
                    if 'error' in result:
                        f.write(f"    Error: {result['error']}\n")
                    
                    f.write("\n")
            
            # Improvements
            if 'base' in results and 'pretrained' in results:
                base_loss = results['base']['avg_loss']
                final_loss = results['pretrained']['avg_loss']
                improvement = ((base_loss - final_loss) / base_loss) * 100
                
                f.write("=" * 70 + "\n")
                f.write("IMPROVEMENT ANALYSIS\n")
                f.write("=" * 70 + "\n")
                f.write(f"Base Model Loss: {base_loss:.4f}\n")
                f.write(f"Pretrained Model Loss: {final_loss:.4f}\n")
                f.write(f"Improvement: {improvement:+.2f}%\n\n")
                
                if improvement > 0:
                    f.write("  Model improved after continual pretraining!\n")
                elif improvement < -5:
                    f.write("  Warning: Model performance degraded significantly!\n")
                    f.write("   Check for overfitting or training issues.\n")
                else:
                    f.write("   Marginal or no improvement detected.\n")
            
            # Initialized vs Base comparison
            if 'base' in results and 'initialized' in results:
                base_loss = results['base']['avg_loss']
                init_loss = results['initialized']['avg_loss']
                init_diff = ((init_loss - base_loss) / base_loss) * 100
                
                f.write("\n" + "=" * 70 + "\n")
                f.write("EMBEDDING INITIALIZATION CHECK\n")
                f.write("=" * 70 + "\n")
                f.write(f"Base Model Loss: {base_loss:.4f}\n")
                f.write(f"Initialized Model Loss: {init_loss:.4f}\n")
                f.write(f"Difference: {init_diff:+.2f}%\n\n")
                
                if abs(init_diff) < 5:
                    f.write("  Embeddings properly initialized (similar to base)\n")
                else:
                    f.write("   Large difference detected - check embedding initialization\n")
        
        logger.info(f"  Summary report saved: {report_path}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--base-model', type=str, default=None,
                       help='Path to base model (HuggingFace ID or local path)')
    parser.add_argument('--initialized-model', type=str, 
                       default='models/initialized_model',
                       help='Path to initialized model')
    parser.add_argument('--pretrained-model', type=str,
                       default='models/checkpoints/chipnemo_final',
                       help='Path to pretrained model')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Max validation samples to evaluate')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(config_path=args.config)
    
    # Run comparison
    results = evaluator.compare_models(
        base_model_path=args.base_model,
        initialized_model_path=args.initialized_model,
        pretrained_model_path=args.pretrained_model,
        max_val_samples=args.max_samples
    )
    
    if not results:
        logger.error("  Evaluation failed - no results")
        return
    
    # Generate plots
    evaluator.plot_comparison(results, save_dir=args.output_dir)
    
    # Save results
    evaluator.save_results(results, save_dir=args.output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("  EVALUATION SUMMARY")
    logger.info("=" * 70)
    
    for key in ['base', 'initialized', 'pretrained']:
        if key in results and results[key]['exists']:
            r = results[key]
            logger.info(f"\n{r['name']}:")
            logger.info(f"  Loss: {r['avg_loss']:.4f}")
            logger.info(f"  Perplexity: {r['perplexity']:.2f}")
            logger.info(f"  Vocab: {r.get('vocab_size', 'N/A'):,}")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"  All results saved to: {args.output_dir}/")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
