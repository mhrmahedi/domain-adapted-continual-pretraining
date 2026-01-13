"""
Simple manual token addition script
Add manually custom domain tokens to the adapted model
This comes after the add_tokenizer.py and before initialize_embeddings.py
"""

from transformers import AutoTokenizer, AutoModel
import torch
import gc
import shutil
from pathlib import Path
import yaml
import sys


_current_file = Path(__file__).resolve()
_project_root = _current_file.parent

# Ensure we're in the project root
if (_project_root / "config").exists():
    project_root = _project_root
elif (_project_root.parent / "config").exists():
    project_root = _project_root.parent
else:
    project_root = Path.cwd()

# Change to project directory
import os
os.chdir(project_root)



def add_manual_tokens_simple():
    """Add manual tokens to existing adapted model"""
    
    print("=" * 80)
    print(" MANUAL TOKEN ADDITION")
    print("=" * 80)
    print()
    
    # ============================================================
    # DEFINE OUR CUSTOM TOKENS HERE
    # ============================================================
    MANUAL_TOKENS = [
        "mahedi",
        "hasan",
        "pytorch",
        "tensorflow",
        "cuda",
        "gpu",
        "transformer",
        "bert",
        "llama",
        # Add more domain-specific tokens as needed
    ]
    # ============================================================
    
    print(f" Tokens to add: {len(MANUAL_TOKENS)}")
    for i, token in enumerate(MANUAL_TOKENS, 1):
        print(f"   {i}. {token}")
    print()
    
    # Check if paths exist
    tokenizer_path = Path("models/tokenizer")
    model_path = Path("models/adapted_model")
    config_path = Path("config/training_config.yaml")
    
    if not tokenizer_path.exists():
        print(f"  Error: Tokenizer not found at {tokenizer_path}")
        print("   Run tokenizer adaptation first!")
        return
    
    if not model_path.exists():
        print(f"  Error: Model not found at {model_path}")
        print("   Run tokenizer adaptation first!")
        return
    
    if not config_path.exists():
        print(f"  Error: Config not found at {config_path}")
        return
    
    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    old_tok_size = len(tokenizer)
    print(f"   Current tokenizer: {old_tok_size:,} tokens")
    print()
    
    # Filter tokens that don't exist yet
    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [t for t in MANUAL_TOKENS if t not in existing_vocab]
    
    if not tokens_to_add:
        print("  All tokens already exist!")
        print("\nTokens already in vocabulary:")
        for token in MANUAL_TOKENS:
            if token in existing_vocab:
                print(f"   ✓ {token}")
        return
    
    print(f"  New tokens to add: {len(tokens_to_add)}")
    for token in tokens_to_add:
        print(f"   + {token}")
    print()
    
    # Add tokens to tokenizer
    print("  Adding tokens to tokenizer...")
    num_added = tokenizer.add_tokens(tokens_to_add)
    print(f"   Tokenizer: {old_tok_size:,} → {len(tokenizer):,} (+{num_added})")
    print()
    
    # Load model
    print("  Loading model...")
    
    # Get base model name from config
    with open(str(config_path), 'r') as f:
        config = yaml.safe_load(f)
    base_model_name = config['model']['base_model']
    
    # Load base tokenizer for initialization
    print(f"   Base model: {base_model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_vocab_size = len(base_tokenizer)
    
    # Load adapted model
    model = AutoModel.from_pretrained(str(model_path))
    old_model_size = model.get_input_embeddings().num_embeddings
    print(f"   Current model: {old_model_size:,} embeddings")
    print()
    
    # Check if sync is needed first
    if old_model_size != old_tok_size:
        print(f"   Warning: Model ({old_model_size}) and tokenizer ({old_tok_size}) were out of sync")
        print("  Resyncing model...")
        model.resize_token_embeddings(old_tok_size, mean_resizing=False)
        
        # Initialize gap
        embeddings = model.get_input_embeddings()
        embedding_weights = embeddings.weight.data
        
        for i in range(old_model_size, old_tok_size):
            avg = torch.mean(embedding_weights[:old_model_size], dim=0)
            embedding_weights[i] = avg
        
        embeddings.weight.data = embedding_weights
        print(f"   Model resynced: {old_model_size:,} → {old_tok_size:,}")
        print()
        old_model_size = old_tok_size
    
    # Resize model for new tokens
    print("  Resizing model...")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    print(f"   Model: {old_model_size:,} → {len(tokenizer):,} embeddings")
    print()
    
    # Initialize new token embeddings
    print("  Initializing new token embeddings...")
    embeddings = model.get_input_embeddings()
    embedding_weights = embeddings.weight.data
    initialized = 0
    
    for token in tokens_to_add:
        token_id = tokenizer.convert_tokens_to_ids(token)
        
        if token_id >= old_model_size:
            # Re-tokenize using base tokenizer
            encoded = base_tokenizer.encode(token, add_special_tokens=False)
            
            if len(encoded) > 0:
                # Average embeddings
                base_embeddings = embedding_weights[encoded]
                avg_embedding = torch.mean(base_embeddings, dim=0)
                embedding_weights[token_id] = avg_embedding
                initialized += 1
            else:
                # Fallback
                avg_embedding = torch.mean(embedding_weights[:base_vocab_size], dim=0)
                embedding_weights[token_id] = avg_embedding
                initialized += 1
    
    embeddings.weight.data = embedding_weights
    
    # Zero-initialize LM head
    if hasattr(model, 'get_output_embeddings'):
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            output_weights = output_emb.weight.data
            for token in tokens_to_add:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id >= old_model_size:
                    output_weights[token_id] = 0.0
            output_emb.weight.data = output_weights
    
    print(f"   Initialized {initialized} embeddings")
    print()
    
    # Save tokenizer
    print("  Saving tokenizer...")
    tokenizer.save_pretrained(str(tokenizer_path))
    print("     Tokenizer saved")
    print()
    
    # Save model
    print("  Saving model...")
    model = model.cpu()
    
    temp_path = Path("models/adapted_model_temp")
    final_path = Path("models/adapted_model")
    
    temp_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(temp_path), safe_serialization=False)
    
    del model
    gc.collect()
    
    if final_path.exists():
        shutil.rmtree(final_path)
    shutil.move(str(temp_path), str(final_path))
    
    print("     Model saved")
    print()
    
    # Summary
    print("=" * 80)
    print("  COMPLETE!")
    print("=" * 80)
    print()
    print("  Summary:")
    print(f"   Tokenizer: {old_tok_size:,} → {len(tokenizer):,} (+{num_added})")
    print(f"   Model: {old_model_size:,} → {len(tokenizer):,} (+{num_added})")
    print()
    print("  New tokens added:")
    for token in tokens_to_add:
        result = tokenizer.tokenize(token)
        print(f"   '{token}' → {result}")
    print()
    print("  Ready for training!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        add_manual_tokens_simple()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
