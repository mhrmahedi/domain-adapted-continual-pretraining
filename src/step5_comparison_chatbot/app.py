"""
Gradio Interface for ChipNeMo Model Comparison
Updated for Base vs Initialized vs Pretrained
"""

import gradio as gr
import time
import logging
from pathlib import Path
from model_loader import ModelManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotApp:
    """Gradio chatbot application"""
    
    def __init__(self, config):
        self.config = config
        self.manager = ModelManager(config)
        
        # Load models
        if not self.manager.load_all_models():
            raise RuntimeError("No models loaded successfully!")
        
        self.loaded_models = self.manager.get_loaded_models()
    
    def generate_comparison(self, prompt, max_tokens, temperature, top_p):
        """Generate responses from all loaded models"""
        if not prompt.strip():
            return self._empty_results()
        
        gen_config = {
            'max_new_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': self.config['generation']['top_k'],
            'do_sample': self.config['generation']['do_sample'],
            'repetition_penalty': self.config['generation']['repetition_penalty']
        }
        
        results = {}
        for model_key in self.loaded_models:
            model_config = self.config['models'][model_key]
            logger.info(f"Generating with {model_config['name']}...")
            
            start_time = time.time()
            response = self.manager.generate(model_key, prompt, gen_config)
            elapsed = time.time() - start_time
            
            word_count = len(response.split())
            tokens_per_sec = word_count / elapsed if elapsed > 0 else 0
            
            results[model_key] = {
                'response': response,
                'time': elapsed,
                'words': word_count,
                'speed': tokens_per_sec
            }
        
        return self._format_results(results)
    
    def _format_results(self, results):
        """Format results for Gradio output"""
        outputs = []
        
        # Add response for each loaded model
        for model_key in ['base', 'adapted', 'trained']:
            if model_key in results:
                r = results[model_key]
                outputs.append(r['response'])
                outputs.append(f"  {r['time']:.2f}s |  {r['words']} words |  {r['speed']:.1f} words/s")
            else:
                outputs.append("Model not loaded")
                outputs.append("")
        
        # Add comparison summary
        if len(results) > 1:
            summary = "  **Comparison Summary:**\n\n"
            for model_key, r in results.items():
                name = self.config['models'][model_key]['name']
                summary += f"- **{name}**: {r['words']} words in {r['time']:.2f}s\n"
        else:
            summary = ""
        
        outputs.append(summary)
        
        return tuple(outputs)
    
    def _empty_results(self):
        """Return empty results"""
        return ("", "", "", "", "", "", "")
    
    def build_interface(self):
        """Build Gradio interface"""
        with gr.Blocks(title=self.config['ui']['title'], theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# {self.config['ui']['title']}")
            gr.Markdown(f"*{self.config['ui']['description']}*")
            
            # Show which models are loaded
            loaded_names = [self.config['models'][k]['name'] for k in self.loaded_models]
            gr.Markdown(f"**Loaded models:** {', '.join(loaded_names)}")
            
            # Input section
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="  Your Question",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
            
            with gr.Row():
                generate_btn = gr.Button("  Generate & Compare", variant="primary", scale=2)
                clear_btn = gr.Button("  Clear", scale=1)
            
            # Settings
            with gr.Accordion("  Generation Settings", open=False):
                with gr.Row():
                    max_tokens = gr.Slider(50, 512, value=self.config['generation']['max_new_tokens'],
                                         label="Max Tokens", step=10)
                    temperature = gr.Slider(0.1, 1.5, value=self.config['generation']['temperature'],
                                          label="Temperature", step=0.1)
                    top_p = gr.Slider(0.1, 1.0, value=self.config['generation']['top_p'],
                                    label="Top-p", step=0.05)
            
            # Output sections
            gr.Markdown("##   Responses")
            
            with gr.Row():
                # Base model
                if 'base' in self.loaded_models:
                    with gr.Column():
                        gr.Markdown(f"###   {self.config['models']['base']['name']}")
                        gr.Markdown(f"*{self.config['models']['base']['description']}*")
                        base_output = gr.Textbox(label="Response", lines=10, interactive=False)
                        base_metrics = gr.Textbox(label="Metrics", lines=1, interactive=False)
                else:
                    base_output = gr.Textbox(visible=False)
                    base_metrics = gr.Textbox(visible=False)
                
                # Adapted model (Initialized)
                if 'adapted' in self.loaded_models:
                    with gr.Column():
                        gr.Markdown(f"###   {self.config['models']['adapted']['name']}")
                        gr.Markdown(f"*{self.config['models']['adapted']['description']}*")
                        adapted_output = gr.Textbox(label="Response", lines=10, interactive=False)
                        adapted_metrics = gr.Textbox(label="Metrics", lines=1, interactive=False)
                else:
                    adapted_output = gr.Textbox(visible=False)
                    adapted_metrics = gr.Textbox(visible=False)
                
                # Trained model (Pretrained)
                if 'trained' in self.loaded_models:
                    with gr.Column():
                        gr.Markdown(f"###   {self.config['models']['trained']['name']}")
                        gr.Markdown(f"*{self.config['models']['trained']['description']}*")
                        trained_output = gr.Textbox(label="Response", lines=10, interactive=False)
                        trained_metrics = gr.Textbox(label="Metrics", lines=1, interactive=False)
                else:
                    trained_output = gr.Textbox(visible=False)
                    trained_metrics = gr.Textbox(visible=False)
            
            # Comparison summary
            comparison = gr.Markdown("*Generate to see comparison*")
            
            # Wire up events
            generate_btn.click(
                fn=self.generate_comparison,
                inputs=[prompt_input, max_tokens, temperature, top_p],
                outputs=[base_output, base_metrics, adapted_output, adapted_metrics,
                        trained_output, trained_metrics, comparison]
            )
            
            clear_btn.click(
                lambda: ("", "", "", "", "", "", "", ""),
                outputs=[prompt_input, base_output, base_metrics, adapted_output,
                        adapted_metrics, trained_output, trained_metrics, comparison]
            )
            
            return demo

def launch_chatbot(config):
    """Launch the chatbot application"""
    app = ChatbotApp(config)
    demo = app.build_interface()
    
    logger.info("\n" + "="*70)
    logger.info("  Chatbot ready!")
    logger.info(f"  Access at: http://localhost:{config['ui']['port']}")
    logger.info("="*70 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=config['ui']['port'],
        share=config['ui']['share']
    )
