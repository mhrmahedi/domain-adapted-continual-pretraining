"""
Step 3a: ChipNeMo Domain-Adaptive Tokenizer - Token Extraction & Expansion Only
NO EMBEDDING INITIALIZATION (done separately in initialize_embeddings.py)
"""

import yaml
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
from transformers import AutoTokenizer
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("tokenizer_adapter")

class TokenizerAdapter:
    """
    ChipNeMo-style tokenizer adapter - VOCABULARY EXPANSION ONLY
    Embedding initialization is handled separately
    """
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        config_path: str = "config/training_config.yaml",
        domain_data_path: str = "data/curated/curated_data.jsonl"
    ):
        """Initialize tokenizer adapter"""
        self.base_model_name = base_model_name
        self.domain_data_path = Path(domain_data_path)
        
        # Load config
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f" Config file not found: {config_path}")
            raise
        
        # Handle config structure
        try:
            if 'model' in self.config and 'tokenizer_adaptation' in self.config['model']:
                self.tokenizer_config = self.config['model']['tokenizer_adaptation']
            elif 'tokenizer_adaptation' in self.config:
                self.tokenizer_config = self.config['tokenizer_adaptation']
            else:
                logger.warning(" tokenizer_adaptation not in config, using defaults")
                self.tokenizer_config = {'enable': True, 'min_frequency': 200}
        except Exception as e:
            logger.warning(f" Config parsing issue: {e}, using defaults")
            self.tokenizer_config = {'enable': True, 'min_frequency': 200}
        
        # Check if enabled
        if not self.tokenizer_config.get('enable', True):
            logger.info(" Tokenizer adaptation is DISABLED in config")
            self.base_tokenizer = None
            return
        
        # Load base tokenizer
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Set padding token if not set
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
                logger.info("  Set pad_token = eos_token")
            
            logger.info(f" Loaded base tokenizer: {base_model_name}")
            logger.info(f" Base vocabulary size: {len(self.base_tokenizer)}")
        except Exception as e:
            logger.error(f" Failed to load tokenizer: {e}")
            raise
        
        self.original_vocab_size = len(self.base_tokenizer)
        
        # Statistics tracking
        self.stats = {
            'original_vocab_size': self.original_vocab_size,
            'total_words': 0,
            'unique_words': 0,
            'candidate_tokens_found': 0,
            'high_freq_candidates': 0,
            'new_tokens_added': 0,
            'final_vocab_size': self.original_vocab_size
        }
    
    def _load_domain_texts(self, sample_limit: int = 500) -> List[str]:
        """Load domain-specific texts from curated data"""
        if not self.domain_data_path.exists():
            logger.warning(f" Domain data not found: {self.domain_data_path}")
            return []
        
        all_text = []
        try:
            with open(self.domain_data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= sample_limit:
                        break
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        if text and len(text.strip()) > 0:
                            all_text.append(text)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f" Loaded {len(all_text)} documents from domain data")
        except Exception as e:
            logger.error(f" Error reading domain data: {e}")
            return []
        
        return all_text
    
    def extract_candidate_tokens(
        self,
        min_frequency: int = 200,
        min_word_length: int = 3
    ) -> List[Dict]:
        """
        Extract candidate tokens filtered by frequency
        Returns list of dicts with: word, frequency, split_count, base_tokens
        """
        if not self.tokenizer_config.get('enable', True):
            return []
        
        logger.info("\n" + "="*70)
        logger.info("  ChipNeMo Domain Token Extraction")
        logger.info("="*70)
        logger.info(f" Minimum frequency threshold: {min_frequency}")
        logger.info(f" Minimum word length: {min_word_length}\n")
        
        # Load domain texts
        texts = self._load_domain_texts()
        if not texts:
            logger.warning(" No domain texts loaded")
            return []
        
        # Extract all words with frequencies
        logger.info(" Extracting vocabulary with frequencies...")
        all_words = []
        for text in texts:
            # Clean and split
            cleaned = text.replace(',', ' ').replace('.', ' ').replace('(', ' ')
            cleaned = cleaned.replace(')', ' ').replace(';', ' ').replace(':', ' ')
            words = cleaned.split()
            
            for word in words:
                word = word.strip().lower()
                if len(word) >= min_word_length and any(c.isalpha() for c in word):
                    all_words.append(word)
        
        # Count frequencies
        word_counts = Counter(all_words)
        self.stats['total_words'] = len(all_words)
        self.stats['unique_words'] = len(word_counts)
        
        logger.info(f" Total words: {len(all_words):,}")
        logger.info(f" Unique words: {len(word_counts):,}")
        
        # Identify candidates (not in base vocab + splits into multiple tokens)
        logger.info("\n🔎 Identifying candidate tokens...")
        base_vocab = set(self.base_tokenizer.get_vocab().keys())
        
        all_candidates = []
        for word, frequency in word_counts.items():
            if word not in base_vocab:
                tokens = self.base_tokenizer.tokenize(word)
                if len(tokens) > 1:  # Only words that split
                    all_candidates.append({
                        'word': word,
                        'frequency': frequency,
                        'split_count': len(tokens),
                        'base_tokens': tokens
                    })
        
        self.stats['candidate_tokens_found'] = len(all_candidates)
        logger.info(f" Total candidates (all frequencies): {len(all_candidates):,}")
        
        # Filter by minimum frequency
        high_freq_candidates = [
            c for c in all_candidates
            if c['frequency'] >= min_frequency
        ]
        
        self.stats['high_freq_candidates'] = len(high_freq_candidates)
        logger.info(f" High-frequency candidates (≥{min_frequency}): {len(high_freq_candidates):,}")
        
        # Sort by frequency descending
        high_freq_candidates.sort(key=lambda x: x['frequency'], reverse=True)
        
        # Display top candidates
        if high_freq_candidates:
            logger.info(f"\n Top 20 most frequent candidates:")
            logger.info("-" * 70)
            for i, item in enumerate(high_freq_candidates[:20], 1):
                logger.info(
                    f"{i:3d}. '{item['word']:25s}' - "
                    f"Freq: {item['frequency']:5d}, "
                    f"Splits: {item['split_count']} → {item['base_tokens']}"
                )
        
        return high_freq_candidates
    
    def add_tokens_to_vocabulary(
        self,
        candidates: List[Dict],
        max_tokens: Optional[int] = None,
        save_path: str = "models/tokenizer_adapted"
    ) -> AutoTokenizer:
        """
        Add candidate tokens to vocabulary (NO embedding initialization)
        
        Args:
            candidates: List of candidate token dicts
            max_tokens: Maximum number of tokens to add (None = no limit)
            save_path: Where to save the adapted tokenizer
            
        Returns:
            Adapted tokenizer
        """
        if not candidates:
            logger.warning(" No candidates to add")
            return self.base_tokenizer
        
        logger.info("\n" + "="*70)
        logger.info(" Adding Tokens to Vocabulary")
        logger.info("="*70)
        
        # Apply max tokens limit
        if max_tokens is not None and len(candidates) > max_tokens:
            candidates = candidates[:max_tokens]
            logger.info(f" Limiting to top {max_tokens} tokens\n")
        
        new_tokens = [c['word'] for c in candidates]
        old_vocab_size = len(self.base_tokenizer)
        
        # Add tokens to vocabulary
        try:
            num_added = self.base_tokenizer.add_tokens(new_tokens)
            self.stats['new_tokens_added'] = num_added
            self.stats['final_vocab_size'] = len(self.base_tokenizer)
            
            logger.info(f" Successfully added {num_added} tokens")
            logger.info(f" Vocabulary size: {old_vocab_size} → {len(self.base_tokenizer)}")
        except Exception as e:
            logger.error(f" Error adding tokens: {e}")
            return self.base_tokenizer
        
        # Save adapted tokenizer
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.base_tokenizer.save_pretrained(save_path)
            logger.info(f"\n Adapted tokenizer saved to: {save_path}")
        except Exception as e:
            logger.error(f" Error saving tokenizer: {e}")
            return self.base_tokenizer
        
        # Save token list with frequencies and metadata
        self._save_token_metadata(candidates, save_path)
        
        # Print statistics
        self._print_statistics()
        
        logger.info("\n" + "="*70)
        logger.info(" Vocabulary Expansion Complete!")
        logger.info("="*70)
        logger.info("  IMPORTANT: Embeddings NOT initialized yet")
        logger.info(" Next step: Run initialize_embeddings.py")
        logger.info("="*70 + "\n")
        
        return self.base_tokenizer
    
    def _save_token_metadata(self, candidates: List[Dict], save_path: Path):
        """Save detailed token metadata"""
        # Save as text file
        token_list_path = save_path / "added_tokens_with_frequency.txt"
        with open(token_list_path, 'w', encoding='utf-8') as f:
            f.write(f"ADDED TOKENS (Total: {len(candidates)})\n")
            f.write("="*70 + "\n\n")
            for i, item in enumerate(candidates, 1):
                f.write(
                    f"{i:4d}. '{item['word']:30s}' - "
                    f"Freq: {item['frequency']:5d}, "
                    f"Splits: {item['split_count']}\n"
                )
        
        logger.info(f" Token list saved to: {token_list_path}")
        
        # Save as JSON for programmatic access
        token_json_path = save_path / "added_tokens_metadata.json"
        with open(token_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_tokens_added': len(candidates),
                'original_vocab_size': self.stats['original_vocab_size'],
                'final_vocab_size': self.stats['final_vocab_size'],
                'tokens': candidates
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Token metadata saved to: {token_json_path}")
    
    def _print_statistics(self):
        """Print adaptation statistics"""
        logger.info("\n" + "="*70)
        logger.info(" Token Extraction Statistics")
        logger.info("="*70)
        logger.info(f"Total words processed:     {self.stats['total_words']:>8,}")
        logger.info(f"Unique words found:        {self.stats['unique_words']:>8,}")
        logger.info(f"All candidates (any freq): {self.stats['candidate_tokens_found']:>8,}")
        logger.info(f"High-freq candidates:      {self.stats['high_freq_candidates']:>8,}")
        logger.info(f"Tokens added:              {self.stats['new_tokens_added']:>8,}")
        logger.info(f"")
        logger.info(f"Original vocabulary:       {self.stats['original_vocab_size']:>8,}")
        logger.info(f"Final vocabulary:          {self.stats['final_vocab_size']:>8,}")
        
        if self.stats['original_vocab_size'] > 0:
            increase = ((self.stats['final_vocab_size'] - self.stats['original_vocab_size'])
                       / self.stats['original_vocab_size'] * 200)
            logger.info(f"Vocabulary increase:       {increase:>7.2f}%")
        logger.info("="*70)


def main():
    """Main execution"""
    logger.info("="*70)
    logger.info(" ChipNeMo Tokenizer Adaptation - Step 1: Vocabulary Expansion")
    logger.info("="*70 + "\n")
    
    # Configuration
    config_path = "config/training_config.yaml"
    base_model = "meta-llama/Meta-Llama-3.1-8B"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f" Config file not found: {config_path}")
        return
    
    # Get settings from config
    try:
        if 'model' in config and 'tokenizer_adaptation' in config['model']:
            is_enabled = config['model']['tokenizer_adaptation'].get('enable', True)
            min_freq = config['model']['tokenizer_adaptation'].get('min_frequency', 200)
        elif 'tokenizer_adaptation' in config:
            is_enabled = config['tokenizer_adaptation'].get('enable', True)
            min_freq = config['tokenizer_adaptation'].get('min_frequency', 200)
        else:
            is_enabled = True
            min_freq = 200
    except Exception as e:
        logger.warning(f" Config parsing error: {e}, using defaults")
        is_enabled = True
        min_freq = 200
    
    if not is_enabled:
        logger.info(" Tokenizer adaptation is DISABLED in config")
        return
    
    # Settings
    MIN_FREQUENCY = min_freq
    MAX_TOKENS = None  # Set to limit number of tokens, e.g., 2000
    
    logger.info(f"  Settings:")
    logger.info(f"   Base model: {base_model}")
    logger.info(f"   Minimum frequency: {MIN_FREQUENCY}")
    logger.info(f"   Maximum tokens: {MAX_TOKENS if MAX_TOKENS else 'No limit'}\n")
    
    # Create adapter
    adapter = TokenizerAdapter(
        base_model_name=base_model,
        config_path=config_path
    )
    
    # Step 1: Extract candidates
    candidates = adapter.extract_candidate_tokens(
        min_frequency=MIN_FREQUENCY
    )
    
    if not candidates:
        logger.warning(" No candidates found. Try lowering min_frequency.")
        return
    
    # Step 2: Add tokens to vocabulary
    adapted_tokenizer = adapter.add_tokens_to_vocabulary(
        candidates=candidates,
        max_tokens=MAX_TOKENS,
        save_path="models/tokenizer_adapted"
    )
    
    logger.info(" Done! Next step: initialize_embeddings.py")


if __name__ == "__main__":
    main()
