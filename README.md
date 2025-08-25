# CURE-Bench Starter Kit

[![ProjectPage](https://img.shields.io/badge/CUREBench-Page-red)](https://curebench.ai) [![ProjectPage](https://img.shields.io/badge/CUREBench-Kaggle-green)](https://www.kaggle.com/competitions/cure-bench) [![Q&A](https://img.shields.io/badge/Question-Answer-blue)](QA.md)

A simple inference framework for the CURE-Bench bio-medical AI competition. This starter kit provides an easy-to-use interface for generating submission data in CSV format.

## Updates
 2025.08.08: **[Question&Answer page](QA.md)**: We have created a Q&A page to share all our responses to questions from participants, ensuring fair competition.

## Quick Start

### Installation Dependencies
```bash
pip install -r requirements.txt
```

### For Unsloth Models (Google Colab)
If using Unsloth optimized models in Google Colab:
```bash
# Install Unsloth (run in Colab)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
```

## Baseline Setup

If you want to use the ChatGPT baseline:
1. Set up your Azure OpenAI resource
2. Configure environment variables:
```bash
export AZURE_OPENAI_API_KEY_O1="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

If you want to use the open-ended models, such as Qwen:
For local models, ensure you have sufficient GPU memory:
```bash
# Install CUDA-compatible PyTorch if needed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transfomers
```

If you want to use Unsloth models:
See the dedicated section below for Unsloth setup.

## 📁 Project Structure

```
├── eval_framework.py              # Main evaluation framework
├── dataset_utils.py               # Dataset loading utilities
├── run.py                         # Command-line evaluation script
├── run_unsloth.py                 # Unsloth-specific evaluation script
├── colab_unsloth_example.py       # Colab example for Unsloth
├── CureBench_Unsloth_Evaluation.ipynb  # Jupyter notebook for Colab
├── metadata_config.json           # Example metadata configuration
├── metadata_config_unsloth.json   # Unsloth-specific configuration
├── requirements.txt               # Python dependencies
└── competition_results/           # Output directory for your results
```

## Dataset Preparation

Download the val and test dataset from the Kaggle site:
```
https://www.kaggle.com/competitions/cure-bench
```

For val set, configure datasets in your `metadata_config_val.json` file with the following structure:
```json
{
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/your/curebench_valset.jsonl",
    "description": "CureBench 2025 val questions"
  }
}
```

For test set, configure datasets in your `metadata_config_test.json` file with the following structure:
```json
{
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/your/curebench_testset.jsonl",
    "description": "CureBench 2025 test questions"
  }
}
```

## Usage Examples

### Basic Evaluation with Config File
```bash
# Run with configuration file (recommended)
python run.py --config metadata_config_test.json
```

### Unsloth Models Evaluation

#### Method 1: Using the Jupyter Notebook (Recommended for Colab)
1. Upload `CureBench_Unsloth_Evaluation.ipynb` to Google Colab
2. Install dependencies and load your Unsloth model
3. Follow the step-by-step evaluation process

#### Method 2: Using Python Script
```bash
# Run Unsloth evaluation with configuration
python run_unsloth.py --config metadata_config_unsloth.json
```

#### Method 3: Using the Framework Directly
```python
from unsloth import FastModel
from eval_framework import CompetitionKit

# Load Unsloth model
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/medgemma-4b-it-bnb-4bit",
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=True,
    full_finetuning=False,
)

# Initialize framework
kit = CompetitionKit(config_path="metadata_config_unsloth.json")
kit.load_model(
    model_name="unsloth/medgemma-4b-it-bnb-4bit",
    model_type="unsloth",
    model_instance=model,
    tokenizer_instance=tokenizer
)

# Run evaluation
results = kit.evaluate("cure_bench_phase_1")
submission_path = kit.save_submission_with_metadata(results=[results])
```

## 🔧 Configuration

### Metadata Configuration
Create a `metadata_config_val.json` file:
```json
{
  "metadata": {
    "model_name": "gpt-4o-1120",
    "model_type": "ChatGPTModel",
    "track": "internal_reasoning",
    "base_model_type": "API",
    "base_model_name": "gpt-4o-1120",
    "dataset": "cure_bench_pharse_1",
    "additional_info": "",
    "average_tokens_per_question": "",
    "average_tools_per_question": "",
    "tool_category_coverage": ""
  },
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/curebench_valset.jsonl",
    "description": "CureBench 2025 val questions"
  },
  "output_dir": "competition_results",
  "output_file": "submission.csv"
}
```

### Unsloth Configuration Example
For Unsloth models, use `metadata_config_unsloth.json`:
```json
{
  "metadata": {
    "model_name": "unsloth/medgemma-4b-it-bnb-4bit",
    "model_type": "UnslothModel",
    "track": "internal_reasoning",
    "base_model_type": "OpenWeighted",
    "base_model_name": "medgemma-4b-it",
    "dataset": "cure_bench_phase_1",
    "additional_info": "Unsloth optimized medical model with 4-bit quantization"
  },
  "dataset": {
    "dataset_name": "cure_bench_phase_1",
    "dataset_path": "curebench_testset_phase1.jsonl",
    "description": "CureBench 2025 Phase 1 test questions"
  },
  "output_dir": "unsloth_results"
}
```

### Required Metadata Fields
- `model_name`: Display name of your model
- `track`: Either "internal_reasoning" or "agentic_reasoning"
- `base_model_type`: Either "API" or "OpenWeighted"
- `base_model_name`: Name of the underlying model
- `dataset`: Name of the dataset

Note: You can leave the following fields empty for the first round of submissions:
`additional_info`,`average_tokens_per_question`, `average_tools_per_question`, and `tool_category_coverage`.
**Please ensure these fields are filled for the final submission.**

### Supported Model Types
The framework supports multiple model types:
1. **ChatGPT/OpenAI**: API-based models (GPT-4, etc.)
2. **Local**: HuggingFace transformers models
3. **Unsloth**: Optimized models with Unsloth library
4. **Custom**: User-defined inference functions

### Question Type Support
The framework handles three distinct question types:
1. **Multiple Choice**: Questions with lettered options (A, B, C, D, E)
2. **Open-ended Multiple Choice**: Open-ended questions converted to multiple choice format  
3. **Open-ended**: Free-form text answers

## Output Format

The framework generates submission files in CSV format with a zip package containing metadata. The CSV structure includes:
- `id`: Question identifier
- `prediction`: Model's answer (choice for multiple choice, text for open-ended)
- `reasoning_trace`: Model's reasoning process
- `choice`: The choice for the multi-choice questions.

The accompanying metadata includes:
```json
{
  "meta_data": {
    "model_name": "gpt-4o-1120",
    "track": "internal_reasoning",
    "model_type": "ChatGPTModel",
    "base_model_type": "API", 
    "base_model_name": "gpt-4o-1120",
    "dataset": "cure_bench_pharse_1",
    "additional_info": "",
    "average_tokens_per_question": "",
    "average_tools_per_question": "",
    "tool_category_coverage": ""
  }
}
```

## Support

For issues and questions: 
1. Check the error messages (they're usually helpful!)
2. Ensure all dependencies are installed
3. Review the examples in this README
4. Open an Github Issue.

Happy competing!
