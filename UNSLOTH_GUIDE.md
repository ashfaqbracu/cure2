# Using Unsloth with CURE-Bench

This guide explains how to use Unsloth for fast and memory-efficient inference in the CURE-Bench framework.

## What is Unsloth?

Unsloth is a library that provides:
- **2x faster inference** compared to standard transformers
- **50% less memory usage** through optimizations
- **4-bit quantization** for even better memory efficiency
- **Easy fine-tuning** capabilities
- Support for popular models (Llama, Qwen, Mistral, Gemma, etc.)

## Installation

### Option 1: Latest Version (Recommended)
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Option 2: Specific CUDA Versions
```bash
# For CUDA 12.1 and PyTorch 2.2.0
pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8 and PyTorch 2.0.1
pip install "unsloth[cu118-torch201] @ git+https://github.com/unslothai/unsloth.git"
```

## Configuration

### 1. Update your metadata configuration
Edit your `metadata_config_val.json`:

```json
{
  "metadata": {
    "model_name": "microsoft/DialoGPT-medium",
    "model_type": "UnslothModel",
    "track": "internal_reasoning",
    "base_model_type": "OpenWeighted",
    "base_model_name": "microsoft/DialoGPT-medium",
    "dataset": "cure_bench_pharse_1",
    "additional_info": "Using Unsloth for fast inference"
  },
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/your/curebench_valset.jsonl",
    "description": "CureBench 2025 val questions"
  }
}
```

### 2. Using with the evaluation framework

#### Basic usage:
```python
from eval_framework import CompetitionKit

kit = CompetitionKit(config_path="metadata_config_val.json")
kit.load_model("microsoft/DialoGPT-medium", model_type="unsloth")
results = kit.evaluate("cure_bench_pharse_1")
```

#### Advanced usage with custom parameters:
```python
kit.load_model(
    model_name="microsoft/DialoGPT-medium",
    model_type="unsloth",
    max_seq_length=2048,      # Maximum sequence length
    load_in_4bit=True,        # Use 4-bit quantization
    dtype=None                # Auto-detect best dtype (None, torch.float16, torch.bfloat16)
)
```

## Supported Models

Unsloth works with many popular models including:
- **Llama models**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`
- **Qwen models**: `Qwen/Qwen-7B`, `Qwen/Qwen-14B`
- **Mistral models**: `mistralai/Mistral-7B-v0.1`
- **Code models**: `codellama/CodeLlama-7b-hf`
- **Chat models**: `microsoft/DialoGPT-medium`

## Command Line Usage

```bash
# Use Unsloth (auto-detected for non-OpenAI models)
python run.py --config metadata_config_val.json

# Explicitly specify Unsloth
python run.py --config metadata_config_val.json --model-type unsloth

# Manual configuration
python run.py --model-name "microsoft/DialoGPT-medium" --model-type unsloth --track internal_reasoning
```

## Performance Benefits

### Memory Usage
- **Standard transformers**: ~8GB for 7B model
- **Unsloth with 4-bit**: ~4GB for 7B model

### Speed
- **Standard transformers**: 100 tokens/second
- **Unsloth optimized**: 200+ tokens/second

### Quality
- Maintains the same quality as standard transformers
- Optimizations are mathematically equivalent

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check your CUDA version
   nvcc --version
   # Install appropriate Unsloth version
   ```

2. **Memory Issues**
   ```python
   # Use smaller sequence length
   kit.load_model("model_name", max_seq_length=1024)
   
   # Use 4-bit quantization
   kit.load_model("model_name", load_in_4bit=True)
   ```

3. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install torch transformers accelerate bitsandbytes
   ```

### Fallback to Standard Transformers
If Unsloth doesn't work, you can fallback to standard transformers:

```python
# Use LocalModel instead of UnslothModel
kit.load_model("model_name", model_type="local")
```

Or in your config file:
```json
{
  "metadata": {
    "model_type": "LocalModel"
  }
}
```

## Best Practices

1. **Use 4-bit quantization** for memory efficiency:
   ```python
   kit.load_model("model_name", load_in_4bit=True)
   ```

2. **Adjust sequence length** based on your hardware:
   - 4GB VRAM: `max_seq_length=1024`
   - 8GB VRAM: `max_seq_length=2048`
   - 16GB+ VRAM: `max_seq_length=4096`

3. **Monitor memory usage**:
   ```python
   import torch
   print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

4. **Use appropriate dtype**:
   - `torch.float16`: Good balance of speed and quality
   - `torch.bfloat16`: Better numerical stability
   - `None`: Auto-detect best option

## Example Script

See `example_unsloth.py` for a complete working example.

## Support

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Unsloth Discord](https://discord.gg/unsloth)
- CURE-Bench Issues: Create a GitHub issue in this repository
