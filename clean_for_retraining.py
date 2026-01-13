"""
clean_for_retraining.py
Cleans old training artifacts and prepares for fresh training
"""

import shutil
import os
from pathlib import Path
import json
import yaml

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    """Print section header"""
    print(f"\n🔹 {text}")
    print("-" * 70)

def check_path_exists(path, name):
    """Check if required path exists"""
    if path.exists():
        print(f"    {name}: Found")
        return True
    else:
        print(f"    {name}: NOT FOUND")
        return False

def delete_directory(path, name):
    """Delete directory and its contents"""
    if path.exists():
        try:
            shutil.rmtree(path)
            print(f"     Deleted: {name}")
            return True
        except Exception as e:
            print(f"    Error deleting {name}: {e}")
            return False
    else:
        print(f"     {name}: Already clean (doesn't exist)")
        return True

def backup_directory(path, backup_suffix="_backup"):
    """Backup directory before deleting"""
    if path.exists():
        backup_path = Path(str(path) + backup_suffix)
        
        # If backup already exists, add timestamp
        if backup_path.exists():
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(str(path) + f"_{timestamp}")
        
        try:
            shutil.copytree(path, backup_path)
            print(f"    Backed up to: {backup_path}")
            return True
        except Exception as e:
            print(f"     Backup failed: {e}")
            return False
    return False

def get_directory_size(path):
    """Calculate directory size in MB"""
    if not path.exists():
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)  # Convert to MB

def check_config_settings(config_path):
    """Check and display current training config"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        training = config.get('training', {})
        lora = config.get('lora', {})
        
        print(f"\n  Current Training Configuration:")
        print(f"  Learning Rate: {training.get('learning_rate', 'N/A')}")
        print(f"  Epochs: {training.get('num_train_epochs', 'N/A')}")
        print(f"  Batch Size: {training.get('per_device_train_batch_size', 'N/A')}")
        print(f"  Gradient Accumulation: {training.get('gradient_accumulation_steps', 'N/A')}")
        print(f"  LoRA Rank: {lora.get('r', 'N/A')}")
        print(f"  LoRA Alpha: {lora.get('lora_alpha', 'N/A')}")
        
        # Recommendations
        lr = training.get('learning_rate', 0)
        epochs = training.get('num_train_epochs', 0)
        
        if lr > 3e-4:
            print(f"\n     WARNING: Learning rate {lr} is high for small datasets")
            print(f"     Recommended: 2e-4 or lower")
        
        if epochs > 7:
            print(f"\n     WARNING: {epochs} epochs may cause overfitting on small data")
            print(f"     Recommended: 5 epochs or fewer")
        
        return True
    except Exception as e:
        print(f"    Error reading config: {e}")
        return False

def verify_required_files():
    """Verify all required files exist before training"""
    print_section("Verifying Required Files")
    
    checks = {
        "Initialized Model": Path("models/initialized_model"),
        "Adapted Tokenizer": Path("models/tokenizer_adapted"),
        "Curated Data": Path("data/curated/curated_data.jsonl"),
        "Training Config": Path("config/training_config.yaml"),
    }
    
    all_good = True
    for name, path in checks.items():
        exists = check_path_exists(path, name)
        if not exists:
            all_good = False
    
    # Check embedding initialization report
    report_path = Path("models/initialized_model/embedding_initialization_report.json")
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            print(f"\n   Embedding Initialization Report:")
            print(f"     New tokens: {report.get('num_new_tokens', 'N/A')}")
            print(f"     Input initialized: {report.get('input_initialized', 'N/A')}")
            print(f"     Input failed: {report.get('input_failed', 'N/A')}")
            print(f"     Output zeroed: {report.get('output_zeroed', 'N/A')}")
            
            verification = report.get('verification', {})
            input_ok = verification.get('input_embeddings', {}).get('all_non_zero', False)
            output_ok = verification.get('output_embeddings', {}).get('all_zero', False)
            
            if input_ok and output_ok:
                print(f"       Embeddings properly initialized")
            else:
                print(f"       Embedding issues detected!")
                all_good = False
        except:
            print(f"     Could not read embedding report")
    
    return all_good

def main():
    """Main cleaning workflow"""
    print_header("  Clean Training Artifacts for Retraining")
    
    # Define paths
    checkpoints_dir = Path("models/checkpoints")
    evaluation_dir = Path("results/evaluation")
    
    # Show current disk usage
    print_section("Current Disk Usage")
    
    if checkpoints_dir.exists():
        size = get_directory_size(checkpoints_dir)
        print(f"  Checkpoints: {size:.2f} MB")
    else:
        print(f"  Checkpoints: Not found (clean)")
    
    if evaluation_dir.exists():
        size = get_directory_size(evaluation_dir)
        print(f"  Evaluation results: {size:.2f} MB")
    else:
        print(f"  Evaluation results: Not found (clean)")
    
    # Ask for backup
    print_section("Backup Options")
    backup = input("  Do you want to backup old checkpoints before deleting? (y/n): ").strip().lower()
    
    if backup == 'y':
        if checkpoints_dir.exists():
            backup_directory(checkpoints_dir)
    
    # Clean checkpoints
    print_section("Cleaning Old Training Artifacts")
    delete_directory(checkpoints_dir, "models/checkpoints/")
    delete_directory(evaluation_dir, "results/evaluation/")
    
    # Recreate empty directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    print(f"    Created empty directories")
    
    # Verify required files
    all_good = verify_required_files()
    
    # Check config
    print_section("Training Configuration")
    check_config_settings(Path("config/training_config.yaml"))
    
    # Final status
    print_header("  Cleaning Complete!")
    
    if all_good:
        print("\n  All systems ready for training!")
        print("\n  Next steps:")
        print("   1. Review training config above")
        print("   2. If settings look good, run:")
        print("      python src/continual_pretraining.py")
        print("\n   3. Monitor training with:")
        print("      tensorboard --logdir models/checkpoints/logs")
    else:
        print("\n  Some required files are missing!")
        print("\n  Required actions:")
        print("   1. If initialized_model missing:")
        print("      python src/initialize_embeddings.py")
        print("\n   2. If data missing:")
        print("      python src/extract_documents.py")
        print("      python src/data_curation.py")
        print("\n   3. Then rerun this cleaning script")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
