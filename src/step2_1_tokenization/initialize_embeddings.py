"""
Step 3b: ChipNeMo Embedding Initialization & Verification
Initializes embeddings for adapted tokenizer and verifies correctness
"""

import torch
import yaml
import json
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("embedding_initializer")


class EmbeddingInitializer:
    """
    ChipNeMo-style embedding initialization:
    1. Input embeddings: Average of subword embeddings
    2. Output embeddings (lm_head): Zero vectors
    """
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        adapted_tokenizer_path: str = "models/tokenizer_adapted"
    ):
        """Initialize embedding initializer"""
        self.base_model_name = base_model_name
        self.adapted_tokenizer_path = Path(adapted_tokenizer_path)
        
        # Load base tokenizer (for subword decomposition)
        logger.info(f" Loading base tokenizer: {base_model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        self.base_vocab_size = len(self.base_tokenizer)
        logger.info(f" Base vocabulary size: {self.base_vocab_size:,}")
        
        # Load adapted tokenizer
        logger.info(f" Loading adapted tokenizer: {adapted_tokenizer_path}")
        self.adapted_tokenizer = AutoTokenizer.from_pretrained(
            str(adapted_tokenizer_path),
            trust_remote_code=True
        )
        self.adapted_vocab_size = len(self.adapted_tokenizer)
        logger.info(f" Adapted vocabulary size: {self.adapted_vocab_size:,}")
        
        # Calculate new tokens
        self.num_new_tokens = self.adapted_vocab_size - self.base_vocab_size
        logger.info(f" New tokens to initialize: {self.num_new_tokens:,}\n")
        
        if self.num_new_tokens == 0:
            logger.warning(" No new tokens found! Check tokenizer adaptation.")
        
        # Load token metadata if available
        self.token_metadata = self._load_token_metadata()
    
    def _load_token_metadata(self) -> Optional[dict]:
        """Load token metadata if available"""
        metadata_path = self.adapted_tokenizer_path / "added_tokens_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def initialize_input_embeddings(
        self,
        model: AutoModelForCausalLM
    ) -> tuple:
        """
        Initialize input embeddings via averaging subword embeddings
        Returns: (initialized_count, failed_count)
        """
        logger.info("="*70)
        logger.info(" Initializing Input Embeddings (Subword Averaging)")
        logger.info("="*70)
        
        input_embeddings = model.get_input_embeddings()
        embedding_weights = input_embeddings.weight.data
        hidden_dim = embedding_weights.shape[1]
        
        initialized_count = 0
        failed_count = 0
        
        # Get list of new tokens
        new_token_ids = list(range(self.base_vocab_size, self.adapted_vocab_size))
        
        logger.info(f"Processing {len(new_token_ids)} new tokens...")
        
        for token_id in new_token_ids:
            # Get token string
            token = self.adapted_tokenizer.convert_ids_to_tokens(token_id)
            
            try:
                # Use BASE tokenizer to get subword decomposition
                subword_ids = self.base_tokenizer.encode(
                    token,
                    add_special_tokens=False
                )
                
                # Filter to only base vocabulary
                subword_ids = [sid for sid in subword_ids if sid < self.base_vocab_size]
                
                if len(subword_ids) > 0:
                    # Average the subword embeddings
                    subword_embeddings = embedding_weights[subword_ids]
                    avg_embedding = torch.mean(subword_embeddings, dim=0)
                    embedding_weights[token_id] = avg_embedding
                    initialized_count += 1
                else:
                    # Fallback: small random initialization
                    embedding_weights[token_id] = torch.randn(
                        hidden_dim,
                        dtype=embedding_weights.dtype,
                        device=embedding_weights.device
                    ) * 0.02
                    failed_count += 1
                    logger.warning(f"   No subwords for token ID {token_id} ('{token}'), using random")
                    
            except Exception as e:
                failed_count += 1
                logger.warning(f"   Failed to initialize token ID {token_id}: {e}")
                continue
        
        # Save back to model
        input_embeddings.weight.data = embedding_weights
        
        logger.info(f"\n Input embeddings initialized:")
        logger.info(f"   Successfully: {initialized_count}/{len(new_token_ids)}")
        logger.info(f"   Failed: {failed_count}/{len(new_token_ids)}")
        
        return initialized_count, failed_count
    
    def initialize_output_embeddings(
        self,
        model: AutoModelForCausalLM
    ) -> int:
        """
        Initialize output embeddings (lm_head) to zero vectors
        Returns: count of zeroed embeddings
        """
        logger.info("\n" + "="*70)
        logger.info(" Initializing Output Embeddings (Zero Vectors)")
        logger.info("="*70)
        
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None:
            logger.warning(" Model has no output embeddings")
            return 0
        
        output_weights = output_embeddings.weight.data
        
        # Zero out all new token rows
        zero_count = 0
        new_token_ids = list(range(self.base_vocab_size, self.adapted_vocab_size))
        
        logger.info(f"Zeroing {len(new_token_ids)} output embedding rows...")
        
        for token_id in new_token_ids:
            output_weights[token_id].fill_(0.0)
            zero_count += 1
        
        # Force synchronization for CUDA
        if output_weights.is_cuda:
            torch.cuda.synchronize()
        
        # Save back to model
        output_embeddings.weight.data = output_weights
        
        logger.info(f" Zeroed {zero_count} output embeddings\n")
        
        return zero_count
    
    def verify_embeddings(
        self,
        model: AutoModelForCausalLM,
        num_samples: int = 10
    ) -> dict:
        """
        Verify embedding initialization
        Returns: dict with verification results
        """
        logger.info("="*70)
        logger.info("🔍 Verifying Embedding Initialization")
        logger.info("="*70)
        
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        
        new_token_ids = list(range(self.base_vocab_size, self.adapted_vocab_size))
        sample_ids = new_token_ids[:min(num_samples, len(new_token_ids))]
        
        results = {
            'input_embeddings': {
                'all_non_zero': True,
                'samples': []
            },
            'output_embeddings': {
                'all_zero': True,
                'samples': []
            }
        }
        
        logger.info(f"\n Checking {len(sample_ids)} sample tokens:\n")
        logger.info(f"{'Token ID':<10} {'Token':<25} {'Input Norm':<15} {'Output Norm':<15} {'Status'}")
        logger.info("-" * 80)
        
        for token_id in sample_ids:
            token = self.adapted_tokenizer.convert_ids_to_tokens(token_id)
            
            # Check input embedding
            input_norm = torch.norm(input_embeddings[token_id]).item()
            input_is_zero = (input_norm < 1e-6)
            
            # Check output embedding
            output_norm = torch.norm(output_embeddings[token_id]).item()
            output_is_zero = (output_norm < 1e-6)
            
            # Determine status
            if not input_is_zero and output_is_zero:
                status = " PASS"
            else:
                status = " FAIL"
                if input_is_zero:
                    results['input_embeddings']['all_non_zero'] = False
                if not output_is_zero:
                    results['output_embeddings']['all_zero'] = False
            
            logger.info(
                f"{token_id:<10} {token:<25} {input_norm:<15.6f} {output_norm:<15.6f} {status}"
            )
            
            results['input_embeddings']['samples'].append({
                'token_id': token_id,
                'token': token,
                'norm': input_norm,
                'is_zero': input_is_zero
            })
            
            results['output_embeddings']['samples'].append({
                'token_id': token_id,
                'token': token,
                'norm': output_norm,
                'is_zero': output_is_zero
            })
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info(" Verification Summary")
        logger.info("="*70)
        
        if results['input_embeddings']['all_non_zero']:
            logger.info(" Input embeddings: ALL initialized (non-zero)")
        else:
            logger.error(" Input embeddings: Some are ZERO (should be non-zero)")
        
        if results['output_embeddings']['all_zero']:
            logger.info(" Output embeddings: ALL zero (correct)")
        else:
            logger.error(" Output embeddings: Some are NON-ZERO (should be zero)")
        
        overall_pass = (results['input_embeddings']['all_non_zero'] and 
                       results['output_embeddings']['all_zero'])
        
        if overall_pass:
            logger.info("\n VERIFICATION PASSED!")
        else:
            logger.error("\n VERIFICATION FAILED - Check embeddings")
        
        logger.info("="*70 + "\n")
        
        return results
    
    def initialize_and_verify(
        self,
        model_path: str,
        save_path: str = "models/initialized_model",
        device_map: str = "auto",
        torch_dtype = torch.bfloat16
    ) -> AutoModelForCausalLM:
        """
        Complete workflow: load model, initialize embeddings, verify, and save
        """
        logger.info("\n" + "="*70)
        logger.info(" ChipNeMo Embedding Initialization Workflow")
        logger.info("="*70 + "\n")
        
        # Step 1: Load model
        logger.info(f" Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        logger.info(f" Model loaded\n")
        
        # Step 2: Resize embeddings
        logger.info(f" Resizing token embeddings: {self.base_vocab_size} → {self.adapted_vocab_size}")
        model.resize_token_embeddings(self.adapted_vocab_size)
        logger.info(f" Embeddings resized\n")
        
        # Step 3: Initialize input embeddings
        input_init, input_failed = self.initialize_input_embeddings(model)
        
        # Step 4: Initialize output embeddings
        output_zeroed = self.initialize_output_embeddings(model)
        
        # Step 5: Verify initialization
        verification_results = self.verify_embeddings(model, num_samples=10)
        
        # Step 6: Save model
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" Saving initialized model to: {save_path}")
        model.save_pretrained(save_path)
        self.adapted_tokenizer.save_pretrained(save_path)
        
        # Save verification results
        results_path = save_path / "embedding_initialization_report.json"
        with open(results_path, 'w') as f:
            json.dump({
                'base_vocab_size': self.base_vocab_size,
                'adapted_vocab_size': self.adapted_vocab_size,
                'num_new_tokens': self.num_new_tokens,
                'input_initialized': input_init,
                'input_failed': input_failed,
                'output_zeroed': output_zeroed,
                'verification': verification_results
            }, f, indent=2)
        
        logger.info(f" Verification report saved to: {results_path}")
        logger.info(f"\n Model and tokenizer saved successfully!")
        
        logger.info("\n" + "="*70)
        logger.info(" Embedding Initialization Complete!")
        logger.info("="*70 + "\n")
        
        return model


def main():
    """Main execution"""
    logger.info("="*70)
    logger.info(" ChipNeMo Embedding Initialization & Verification")
    logger.info("="*70 + "\n")
    
    # Configuration
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
    ADAPTED_TOKENIZER_PATH = "models/tokenizer_adapted"
    MODEL_TO_INITIALIZE = "meta-llama/Meta-Llama-3.1-8B"  # Can be base or partially adapted
    SAVE_PATH = "models/initialized_model"
    
    # Check if adapted tokenizer exists
    if not Path(ADAPTED_TOKENIZER_PATH).exists():
        logger.error(f" Adapted tokenizer not found: {ADAPTED_TOKENIZER_PATH}")
        logger.info(" Run add_tokenizer.py first")
        return
    
    # Initialize
    initializer = EmbeddingInitializer(
        base_model_name=BASE_MODEL,
        adapted_tokenizer_path=ADAPTED_TOKENIZER_PATH
    )
    
    if initializer.num_new_tokens == 0:
        logger.error(" No new tokens to initialize")
        return
    
    # Run initialization workflow
    try:
        model = initializer.initialize_and_verify(
            model_path=MODEL_TO_INITIALIZE,
            save_path=SAVE_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        logger.info("  All done! Model ready for continual pretraining.")
        
    except Exception as e:
        logger.error(f" Initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
