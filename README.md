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

## Baseline Setup

If you want to use the ChatGPT baseline:
1. Set up your Azure OpenAI resource
2. Configure environment variables:
```bash
export AZURE_OPENAI_API_KEY_O1="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

If you want to use Unsloth (recommended for local models):
Unsloth provides fast inference and memory-efficient fine-tuning. Install with:
```bash
# For newer systems (recommended)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# For specific CUDA versions
# pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"  # CUDA 12.1, PyTorch 2.2.0
# pip install "unsloth[cu118-torch201] @ git+https://github.com/unslothai/unsloth.git"  # CUDA 11.8, PyTorch 2.0.1
```

If you want to use standard transformers (alternative to Unsloth):
For local models using standard transformers, ensure you have sufficient GPU memory:
```bash
# Install CUDA-compatible PyTorch if needed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
```

## 📁 Project Structure

```
├── eval_framework.py      # Main evaluation framework
├── dataset_utils.py       # Dataset loading utilities
├── run.py                 # Command-line evaluation script
├── example_unsloth.py     # Example using Unsloth for fast inference
├── metadata_config.json   # Example metadata configuration
├── UNSLOTH_GUIDE.md      # Detailed guide for using Unsloth
├── requirements.txt       # Python dependencies
└── competition_results/   # Output directory for your results
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

### Using Different Model Types
```bash
# Use Unsloth (recommended for local models - faster and more memory efficient)
python run.py --config metadata_config_val.json --model-type unsloth

# Use standard transformers 
python run.py --config metadata_config_val.json --model-type local

# Use ChatGPT/OpenAI models
python run.py --config metadata_config_test.json --model-type chatgpt
```

### Manual Model Configuration
```bash
# Specify model parameters manually
python run.py --model-name "microsoft/DialoGPT-medium" --model-type unsloth --track internal_reasoning
```

## 🔧 Configuration

### Metadata Configuration
Create a `metadata_config_val.json` file:
```json
{
  "metadata": {
    "model_name": "microsoft/DialoGPT-medium",
    "model_type": "UnslothModel",
    "track": "internal_reasoning",
    "base_model_type": "OpenWeighted",
    "base_model_name": "microsoft/DialoGPT-medium",
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

### Required Metadata Fields
- `model_name`: Display name of your model
- `track`: Either "internal_reasoning" or "agentic_reasoning"
- `base_model_type`: Either "API" or "OpenWeighted"
- `base_model_name`: Name of the underlying model
- `dataset`: Name of the dataset

Note: You can leave the following fields empty for the first round of submissions:
`additional_info`,`average_tokens_per_question`, `average_tools_per_question`, and `tool_category_coverage`.
**Please ensure these fields are filled for the final submission.**


### Question Type Support
The framework handles three distinct question types:
1. **Multiple Choice**: Questions with lettered options (A, B, C, D, E)
2. **Open-ended Multiple Choice**: Open-ended questions converted to multiple choice format  
3. **Open-ended**: Free-form text answers

### Model Types
The framework supports multiple model types:

1. **UnslothModel** (Recommended for local models):
   - Fast inference using Unsloth optimization
   - Memory efficient with 4-bit quantization by default
   - Supports most popular models (Llama, Qwen, Mistral, etc.)
   - Best for fine-tuning and inference

2. **LocalModel** (Standard transformers):
   - Uses HuggingFace transformers directly
   - More compatible but slower than Unsloth
   - Uses 8-bit quantization by default

3. **ChatGPTModel** (API-based):
   - For OpenAI/Azure OpenAI models
   - Requires API credentials
   - Good for baselines and comparison

4. **CustomModel**:
   - For user-defined models and inference functions
   - Maximum flexibility


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

For detailed Unsloth usage, see [UNSLOTH_GUIDE.md](UNSLOTH_GUIDE.md).

For issues and questions: 
1. Check the error messages (they're usually helpful!)
2. Review the Unsloth guide for troubleshooting
3. Ensure all dependencies are installed
4. Review the examples in this README
5. Open a Github Issue.

Happy competing!
