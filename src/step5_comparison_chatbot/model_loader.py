"""
Model Loader - Handles loading and managing multiple models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage multiple models for comparison"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.device = config['system']['device']
        self.dtype = getattr(torch, config['system']['dtype'])
        
    def load_all_models(self):
        """Load all enabled models"""
        logger.info("Loading models...")
        
        for model_key, model_config in self.config['models'].items():
            if not model_config.get('enabled', True):
                logger.info(f"  Skipping {model_key} (disabled)")
                continue
            
            success = self.load_model(model_key)
            if not success:
                logger.warning(f"  Failed to load {model_key}")
        
        loaded = list(self.models.keys())
        logger.info(f" Loaded {len(loaded)} models: {', '.join(loaded)}")
        
        return len(loaded) > 0
    
    def load_model(self, model_key):
        """Load a single model"""
        model_config = self.config['models'][model_key]
        model_path = model_config['path']
        
        logger.info(f"\n  Loading {model_config['name']}...")
        logger.info(f"   Path: {model_path}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"    Tokenizer: {len(tokenizer)} tokens")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            model.eval()
            
            # Store
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
            
            # Log memory
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                logger.info(f"    Model loaded ({mem_gb:.2f}GB GPU)")
            else:
                logger.info(f"    Model loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"    Failed: {e}")
            return False
    
    def generate(self, model_key, prompt, gen_config):
        """Generate text from specified model"""
        if model_key not in self.models:
            return f"Model '{model_key}' not loaded"
        
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = inputs.to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_config['max_new_tokens'],
                    temperature=gen_config['temperature'],
                    top_p=gen_config['top_p'],
                    top_k=gen_config['top_k'],
                    do_sample=gen_config['do_sample'],
                    repetition_penalty=gen_config['repetition_penalty'],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            input_length = inputs.input_ids.shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {e}"
    
    def get_loaded_models(self):
        """Get list of loaded model keys"""
        return list(self.models.keys())
    
    def get_model_info(self, model_key):
        """Get information about a model"""
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        config = self.config['models'][model_key]
        
        return {
            'name': config['name'],
            'description': config['description'],
            'vocab_size': len(tokenizer),
            'parameters': sum(p.numel() for p in model.parameters()),
            'device': str(model.device)
        }
