"""
Comparison Chatbot - Python Entry Point
Run: python src/step5_comparison_chatbot/run_chatbot.py
"""

import sys
from pathlib import Path
import yaml
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.step5_comparison_chatbot.app import launch_chatbot


def main():
    parser = argparse.ArgumentParser(description="ChipNeMo Comparison Chatbot")
    parser.add_argument(
        "--config",
        type=str,
        default="config/chatbot_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["base", "adapted", "trained"],
        default=["base", "adapted", "trained"],
        help="Which models to load (default: all)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = project_root / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with CLI args
    if args.port:
        config['ui']['port'] = args.port
    if args.share:
        config['ui']['share'] = True
    
    # Filter models
    for model_key in ['base', 'adapted', 'trained']:
        if model_key not in args.models:
            config['models'][model_key]['enabled'] = False
    
    # Launch
    print("="*70)
    print("  ChipNeMo Comparison Chatbot")
    print("="*70)
    print(f"\n  Configuration:")
    print(f"   Config: {config_path}")
    print(f"   Port: {config['ui']['port']}")
    print(f"   Models: {', '.join(args.models)}")
    print(f"   Share: {config['ui']['share']}")
    print()
    
    launch_chatbot(config)


if __name__ == "__main__":
    main()
