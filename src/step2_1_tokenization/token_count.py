"""
Token Counter for Curated Data
Counts total tokens in your dataset using our adpted tokenizer
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np


_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import setup_logger


logger = setup_logger("token_counter")


class TokenCounter:
    """Count tokens in curated dataset"""
    
    def __init__(
        self,
        tokenizer_path: str = "models/tokenizer",
        data_path: str = "data/curated/curated_data.jsonl",
        use_base_tokenizer: bool = False,
        base_model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    ):
        """
        Initialize token counter
        
        Args:
            tokenizer_path: Path to adapted tokenizer (or base model name)
            data_path: Path to curated JSONL data
            use_base_tokenizer: If True, use base tokenizer instead of adapted
            base_model_name: Base model name if using base tokenizer
        """
        self.data_path = Path(data_path)
        self.use_base_tokenizer = use_base_tokenizer
        
        # Load tokenizer
        logger.info("  Loading tokenizer...")
        if use_base_tokenizer:
            logger.info(f"   Using base tokenizer: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            tokenizer_path = Path(tokenizer_path)
            if tokenizer_path.exists():
                logger.info(f"   Using adapted tokenizer: {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                logger.warning(f"   Adapted tokenizer not found at {tokenizer_path}")
                logger.info(f"   Falling back to base tokenizer: {base_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        logger.info(f"   Tokenizer vocab size: {len(self.tokenizer):,}")
        logger.info("")
    
    def count_tokens_in_text(self, text: str) -> int:
        """Count tokens in a single text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def count_tokens_in_dataset(self) -> Dict:
        """
        Count tokens in entire dataset
        
        Returns:
            Dictionary with statistics
        """
        if not self.data_path.exists():
            logger.error(f"  Data file not found: {self.data_path}")
            return {}
        
        logger.info("  Counting tokens in dataset...")
        logger.info(f"   Data file: {self.data_path}")
        logger.info("")
        
        # Statistics
        stats = {
            'total_documents': 0,
            'total_tokens': 0,
            'total_characters': 0,
            'token_counts': [],
            'char_counts': [],
        }
        
        # Read and count
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info(f"  Processing {len(lines):,} documents...")
        logger.info("")
        
        for line in tqdm(lines, desc="Counting tokens"):
            try:
                doc = json.loads(line)
                text = doc.get('text', '') or doc.get('content', '')
                
                if not text:
                    continue
                
                # Count tokens
                token_count = self.count_tokens_in_text(text)
                char_count = len(text)
                
                stats['total_documents'] += 1
                stats['total_tokens'] += token_count
                stats['total_characters'] += char_count
                stats['token_counts'].append(token_count)
                stats['char_counts'].append(char_count)
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error processing document: {e}")
                continue
        
        # Calculate statistics
        if stats['token_counts']:
            stats['avg_tokens_per_doc'] = np.mean(stats['token_counts'])
            stats['median_tokens_per_doc'] = np.median(stats['token_counts'])
            stats['min_tokens'] = np.min(stats['token_counts'])
            stats['max_tokens'] = np.max(stats['token_counts'])
            stats['std_tokens'] = np.std(stats['token_counts'])
            
            stats['avg_chars_per_doc'] = np.mean(stats['char_counts'])
            stats['compression_ratio'] = stats['total_characters'] / stats['total_tokens']
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """Print formatted statistics"""
        if not stats:
            return
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("  TOKEN COUNT STATISTICS")
        logger.info("=" * 80)
        logger.info("")
        
        # Basic stats
        logger.info("  Document Statistics:")
        logger.info(f"   Total documents:        {stats['total_documents']:,}")
        logger.info("")
        
        # Token stats
        logger.info("  Token Statistics:")
        logger.info(f"   Total tokens:           {stats['total_tokens']:,}")
        logger.info(f"   Average tokens/doc:     {stats['avg_tokens_per_doc']:,.1f}")
        logger.info(f"   Median tokens/doc:      {stats['median_tokens_per_doc']:,.1f}")
        logger.info(f"   Min tokens:             {stats['min_tokens']:,}")
        logger.info(f"   Max tokens:             {stats['max_tokens']:,}")
        logger.info(f"   Std deviation:          {stats['std_tokens']:,.1f}")
        logger.info("")
        
        # Character stats
        logger.info("  Character Statistics:")
        logger.info(f"   Total characters:       {stats['total_characters']:,}")
        logger.info(f"   Average chars/doc:      {stats['avg_chars_per_doc']:,.1f}")
        logger.info(f"   Compression ratio:      {stats['compression_ratio']:.2f} chars/token")
        logger.info("")
        
        # Training estimates
        logger.info("🎓 Training Estimates:")
        
        # Estimate for different batch sizes and sequence lengths
        seq_lengths = [512, 1024, 2048, 4096]
        batch_sizes = [1, 2, 4, 8]
        
        for seq_len in seq_lengths:
            num_sequences = stats['total_tokens'] // seq_len
            logger.info(f"   Seq length {seq_len:,}:")
            logger.info(f"      Total sequences:     {num_sequences:,}")
            
            for batch_size in batch_sizes:
                num_batches = num_sequences // batch_size
                logger.info(f"      Batch size {batch_size}:        {num_batches:,} batches")
            logger.info("")
        
        # Token percentiles
        logger.info("  Token Distribution (percentiles):")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(stats['token_counts'], p)
            logger.info(f"   {p}th percentile:        {value:,.0f} tokens")
        
        logger.info("")
        logger.info("=" * 80)
    
    def save_statistics(self, stats: Dict, output_path: str = "data/curated/token_stats.json"):
        """Save statistics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types
        stats_serializable = {
            'total_documents': int(stats['total_documents']),
            'total_tokens': int(stats['total_tokens']),
            'total_characters': int(stats['total_characters']),
            'avg_tokens_per_doc': float(stats['avg_tokens_per_doc']),
            'median_tokens_per_doc': float(stats['median_tokens_per_doc']),
            'min_tokens': int(stats['min_tokens']),
            'max_tokens': int(stats['max_tokens']),
            'std_tokens': float(stats['std_tokens']),
            'avg_chars_per_doc': float(stats['avg_chars_per_doc']),
            'compression_ratio': float(stats['compression_ratio']),
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        
        logger.info(f"  Statistics saved to: {output_path}")


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("  TOKEN COUNTER FOR CURATED DATA")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Initialize counter
        counter = TokenCounter(
            tokenizer_path="models/tokenizer",
            data_path="data/curated/curated_data.jsonl",
            use_base_tokenizer=False,  # Set to True to use base tokenizer
            base_model_name="meta-llama/Meta-Llama-3.1-8B"
        )
        
        # Count tokens
        stats = counter.count_tokens_in_dataset()
        
        if stats:
            # Print statistics
            counter.print_statistics(stats)
            
            # Save statistics
            counter.save_statistics(stats)
            
            logger.info("")
            logger.info("  Token counting complete!")
        else:
            logger.error("  Failed to count tokens")
            return False
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n  Interrupted by user")
        return False
    except Exception as e:
        logger.error(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
