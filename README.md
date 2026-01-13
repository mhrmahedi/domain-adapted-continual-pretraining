
# ChipNeMo: Domain-Adaptive Pretraining for LLaMA 3.1 8B

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)

Implementation of ChipNeMo domain-adaptive continual pretraining methodology for LLaMA 3.1 8B, specialized for chip design and systems engineering domains.

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

##  Overview

This project implements the [ChipNeMo](https://arxiv.org/abs/2311.00176) methodology from NVIDIA Research, applying domain-adaptive continual pretraining to Meta's LLaMA 3.1 8B model. The goal is to enhance the model's understanding of Aircraft, SysML, MBSE, and systems engineering domains.

### Key Achievements

 Successfully extended LLaMA vocabulary with 449 domain-specific tokens  
 Implemented ChipNeMo embedding initialization (input: averaged, output: zero)  
 Created 2-way comparison chatbot (Base vs Adapted Pre-Trained)  
 Complete end-to-end pipeline from data extraction to deployment  

##  Features

- **Domain-Adaptive Tokenization**: Automatically extracts and adds frequent domain terms to vocabulary
- **ChipNeMo Embedding Initialization**: Proper initialization following research paper
- **Memory-Efficient Training**: QLoRA 4-bit quantization for training on consumer GPUs
- **Interactive Evaluation**: Gradio-based comparison interface
- **Modular Pipeline**: Each step is independent and configurable
- **Comprehensive Logging**: Detailed logs for debugging and analysis

##  Project Structure
```
LLAMA-3.1-8B/
│
├── config/ # Configuration files
│ ├── chatbot_config.yaml # Chatbot interface settings
│ ├── data_config.yaml # Data processing config
│ ├── training_config.yaml # Training hyperparameters
│ └── training_config_backup.yaml # Backup configuration
│
├── data/ # Data directory
│ ├── curated/ # Cleaned training data
│ ├── extracted/ # Extracted raw 
│ ├── processed/ # Tokenized datasets
│ └── raw/ # Original documents
│
├── models/ # Model storage
│ ├── checkpoints/ # Training checkpoints
│ │ ├── checkpoints_YYYYMMDD_HHMMSS/ # Timestamped checkpoints
│ │ └── checkpoints_backup/ # Backup checkpoints
│ ├── initialized_model/ # Model after vocab extension
│ └── tokenizer_adapted/ # Extended tokenizer
│
├── results/ # Evaluation results
│ └── evaluation/ # Metrics and comparisons
│
├── src/ # Source code
│ ├── step1_data_extraction/ # Data extraction module
│ │ ├── init.py
│ │ ├── extract_documents.py # Main extraction script
│ │ ├── extract_nougat.py # PDF extraction with Nougat
│ │ └── extract_unstructured.py # Alternative extraction
│ │
│ ├── step2_0_data_curation/ # Data cleaning module
│ │ ├── init.py
│ │ └── data_curation.py # Clean and prepare data
│ │
│ ├── step2_1_tokenization/ # Tokenization module
│ │ ├── init.py
│ │ ├── add_tokenizer.py # ChipNeMo tokenizer adaptation
│ │ ├── initialize_embeddings.py # Embedding initialization
│ │ ├── manual_tokenizer.py # Manual utilities
│ │ └── token_count.py # Token counting
│ │
│ ├── step3_pretraining/ # Training module
│ │ ├── init.py
│ │ └── continual_pretraining.py # DAPT training
│ │
│ ├── step4_evaluation/ # Evaluation module
│ │ ├── init.py
│ │ └── evaluate_models.py # Model evaluation
│ │
│ ├── step5_comparison_chatbot/ # Interactive interface
│ │ ├── init.py
│ │ ├── run_chatbot.py # Launch script
│ │ ├── app.py # Gradio application
│ │ ├── model_loader.py # Model management
│ │ └── metrics.py # Performance metrics
│ │
│ └── utils/ # Utility functions
│ ├── init.py
│ └── logger.py # Logging utilities
│
├── init.py
├── clean_for_retraining.py # Cleanup script
├── results.zip # Archived results
└── README.md # This file

```

##  Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on RTX A6000)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.12+
- **CUDA**: 11.8+ or 12.1+
- **Conda**: For environment management

##  Installation

### 1. Clone Repository

cd ~/Projects
git clone <repository-url>
cd LLAMA-3.1-8B

### 2.0.1 Create Conda Environment
*Step 1: Activate your conda environment*
conda activate llama_env

*Step 2: Install PyTorch with CUDA (via conda)*
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

*Step 3: Install remaining packages from requirements.txt*
pip install -r requirements.txt

### 2.0.2 Alternate Manually Create Conda Environment

Create environment
conda create -n llama_env python=3.12 -y
conda activate llama_env

Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

Install dependencies
pip install transformers accelerate datasets
pip install bitsandbytes peft
pip install gradio pyyaml sentencepiece protobuf



### 3. HuggingFace Authentication

Login to HuggingFace
huggingface-cli login

Paste your token when prompted


### 4. Verify Installation

python << 'EOF'
import torch
import transformers
print(f"✅ PyTorch: {torch.version}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"✅ Transformers: {transformers.version}")
EOF



##  Quick Start

### Complete Pipeline

Activate environment
conda activate llama_env
cd ~/Projects/LLAMA-3.1-8B

Step 1: Extract documents
python src/step1_data_extraction/extract_documents.py

Step 2: Curate data
python src/step2_0_data_curation/data_curation.py

Step 3: Adapt tokenizer
python src/step2_1_tokenization/add_tokenizer.py
python src/step2_1_tokenization/initialize_embeddings.py

Step 4: Train model
python src/step3_pretraining/continual_pretraining.py

Step 5: Model Evaluation
python src/step4_evaluation/evaluate_models.py

Step 6: Launch chatbot
python src/step5_comparison_chatbot/run_chatbot.py
 *Note: Close the terminal with Ctrl + C*

Step 7: Clean adapted model weight for retraining
python clean_for_retraining.py
*Note: For again continual pre training*

### Individual Steps

Run only tokenization
python src/step2_1_tokenization/add_tokenizer.py

Run only training
python src/step3_pretraining/continual_pretraining.py

Launch chatbot with specific models
python src/step5_comparison_chatbot/run_chatbot.py --models base adapted





## We start with giving 35 SAE(ARP) data to our `data/raw/` folder.

##  Pipeline Steps

### Step 1: Data Extraction

**Purpose**: Extract  from domain-specific documents

python src/step1_data_extraction/extract_documents.py (Active Version)
python src/step1_data_extraction/extract_nougat.py (Need to fix librar error for linux version)
python src/step1_data_extraction/extract_documents.py (Some bug in Linux version)



**Input**: `data/raw/*.pdf`  
**Output**:   `data/extracted/*.json`
**Processing**: OCR,  cleaning, metadata extraction

### Step 2: Data Curation

**Purpose**: Clean and prepare data for training

python src/step2_0_data_curation/data_curation.py



**Input**: `data/extracted/*.json`  
**Output**: `data/curated/curated_data.jsonl`  
**Processing**: Deduplication, formatting, train/val split

### Step 3: Tokenizer Adaptation

**Purpose**: Extend vocabulary with domain-specific terms

python src/step2_1_tokenization/add_tokenizer.py


**Process**:
1. Extract frequent domain terms (threshold: 200+ occurrences)
2. Add new tokens to base LLaMA tokenizer

**Manually Add Token**:

python src/step2_1_tokenization/manual_tokenizer.py

4. Count total token: 4.6 Million in our data 

**Input**: `data/curated/curated_data.jsonl` + base tokenizer  
**Output**: 
- `models/tokenizer_adapted/`: Extended tokenizer (128,256 → 128,705 tokens)
- `models/initialized_model/`: Model with initialized embeddings

**Initialize embeddings**:

python src/step2_1_tokenization/initialize_embeddings.py

3. Initialize embeddings to model:
   - **Input embeddings**: Average of subword embeddings
   - **Output embeddings**: Zero vectors (ChipNeMo spec)


Expected:
 ALL CHECKS PASSED - Token initialization is correct!
Input embeddings initialized: 449/449
Output embeddings zeroed: 449/449

**Token Count**:

python src/step2_1_tokenization/token_count.py

4. Count total token: 4.6 Million in our data 

### Step 4: Continual Pretraining

**Purpose**: Train model on domain data using QLoRA

python src/step3_pretraining/continual_pretraining.py


**Configuration**: `config/training_config.yaml`

**Features**:
- 4-bit quantization (QLoRA) for memory efficiency
- LoRA adapters on attention layers (r=32, alpha=64)
- Automatic checkpointing every 20 steps
- Training metrics logging

**Output**: `models/checkpoints/checkpoints_YYYYMMDD_HHMMSS/`

**Monitoring**:
View training progress
tail -f models/checkpoints/*/logs/training.log

### Step 5: Model Comparison

**Purpose**: IQuantitative Comparison of Training Loss and Perplexity between Base vs Adapted vs Trained models

python ssrc/step4_evaluation/evaluate_models.py

**Output**: `results`


**Features**:
- Side-by-side Model Loss and Perplexity comparison
- Tokenizer count

### Step 6: Gradio Chatbot Model Comparison

**Purpose**: Interactive comparison of Base vs Adapted(Skkiped) vs Trained models

python src/step5_comparison_chatbot/run_chatbot.py

 *** Important Note: Do not forget to turn of this code using Ctrl + C ***


**Access**: Open browser at `http://localhost:7860`

**Features**:
- Side-by-side response comparison
- Adjustable generation parameters
- Performance metrics (time, tokens/sec)
- Export results

**Models Compared**:
1. **Base LLaMA 3.1 8B**: Original pretrained model
2. **Adapted**: Extended vocabulary + initialized embeddings (not trained)
3. **Trained**: After continual pretraining on domain data

### Step 7: Clear the current model


**Purpose**: Clear the weights, parameters of the current adapted model for retraining again

python clean_for_retraining.py

 *** Important Note: Do not forget before training the model again ***

##  Configuration

### Data Configuration (`config/data_config.yaml`)

data:
input_dir: "data/extracted"
output_dir: "data/curated"
min_text_length: 100
max_text_length: 100000
remove_duplicates: true
train_split: 0.90
train_split: 0.10



### Training Configuration (`config/training_config.yaml`)

model:
base_model: "meta-llama/Meta-Llama-3.1-8B"
tokenizer_adaptation:
enable: true
min_frequency: 200

training:
output_dir: "models/checkpoints"
learning_rate: 5.0e-4  # ChipNeMo it was 5.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.05  # More warmup for stability
weight_decay: 0.05  # INCREASED from 0.01 (prevent overfitting)
max_grad_norm: 1.0  # More aggressive clipping
save_steps: 20
bf16: true
optim: "paged_adamw_8bit"

lora:
r: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

quantization:
load_in_4bit: true
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_quant_type: "nf4"



### Chatbot Configuration (`config/chatbot_config.yaml`)

models:
base:
name: "Base LLaMA 3.1 8B"
path: "meta-llama/Meta-Llama-3.1-8B"
enabled: true

adapted:
name: "Adapted (Initialized)"
path: "models/initialized_model"
enabled: false (We made this off with toggle )


trained:
name: "Trained (DAPT)"
path: "models/checkpoints/checkpoints_YYYYMMDD_HHMMSS"
enabled: true

generation:
max_new_tokens: 256
temperature: 0.1
top_p: 0.9
do_sample: true

ui:
port: 7860
share: false



##  Results

### Quantitative Metrics

| Metric | Base Model | Adapted Model | Trained Model |
|--------|-----------|---------------|---------------|
| Vocabulary Size | 128,256 | 128,705 (+449) | 128,705 |
| Parameters | 8B | 8B + LoRA | 8B + LoRA |
| Training Loss | 1.57 | 3.62 | 1.39 |


### Key Findings

** Successes**:
- Successfully extended vocabulary with 449 domain-specific tokens
- Proper ChipNeMo embedding initialization verified
- Functional 3-model comparison system

** Challenges**:
- Smaller Data Size
- Trained model underperforms due to limited training corpus
- Validates need for large-scale domain data (millions of tokens)

** Data Insights**:
- Training data: ~34 samples (4.5 tokens)
- Required for success: 1000+ samples (50M-500 tokens)
- ChipNeMo used: Billions of tokens





##  Troubleshooting

### Common Issues


#### Out of Memory

**Symptom**: CUDA OOM error

**Solution** (in `config/training_config.yaml`):
training:
per_device_train_batch_size: 1
gradient_accumulation_steps: 128 # Increase this



#### Model Not Found

**Symptom**: `OSError: model path not valid`

**Solution**:
Check model exists
ls -la models/initialized_model/

Run from project root
cd ~/Projects/LLAMA-3.1-8B
python src/step5_comparison_chatbot/run_chatbot.py



### Development Setup

Clone and install dev dependencies
git clone <repo-url>
cd LLAMA-3.1-8B
pip install -r requirements_dev.txt

Run tests
pytest tests/

Check code style
black src/
flake8 src/



##  References

### Papers

1. **ChipNeMo**: Liu et al., "ChipNeMo: Domain-Adapted LLMs for Chip Design", NVIDIA Research, 2023
   - [arXiv:2311.00176](https://arxiv.org/abs/2311.00176)

2. **LLaMA**: Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", Meta AI, 2023

3. **QLoRA**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023

4. **DAPT**: Gururangan et al., "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks", ACL 2020

### Code & Tools

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Gradio](https://gradio.app/)

### Models

- [LLaMA 3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)

##  License

This project is for academic and research purposes. See [LICENSE](LICENSE) for details.




##  Contact

For questions or issues:
- **Email**: [ratulmdmahedi@gmail.com]


##  Acknowledgments

- NVIDIA Research for the ChipNeMo methodology
- Meta AI for LLaMA 3.1 8B
- HuggingFace team for Transformers library
- BTU Cottbus-Senftenberg for computational resources

---

**Last Updated**: November 27, 2025  
**Version**: 1.0.0  
**Status**: Complete Implementation

---

