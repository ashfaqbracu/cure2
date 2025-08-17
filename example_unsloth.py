#!/usr/bin/env python3
"""
Example script demonstrating Unsloth usage with CURE-Bench

This script shows how to:
1. Load a model using Unsloth for fast inference
2. Run evaluation on the dataset
3. Generate submission files

Requirements:
- Install unsloth: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
- Download the dataset from Kaggle
"""

from eval_framework import CompetitionKit

def main():
    print("🚀 CURE-Bench with Unsloth Example")
    print("=" * 50)
    
    # Initialize the competition kit
    kit = CompetitionKit(config_path="metadata_config_val.json")
    
    # Load model using Unsloth (faster than standard transformers)
    model_name = "microsoft/DialoGPT-medium"  # Example model - replace with your preferred model
    print(f"Loading model: {model_name} with Unsloth...")
    
    # Load model with Unsloth optimizations
    kit.load_model(
        model_name=model_name,
        model_type="unsloth",  # Use Unsloth for fast inference
        max_seq_length=2048,   # Maximum sequence length
        load_in_4bit=True,     # Use 4-bit quantization for memory efficiency
        dtype=None             # Auto-detect best dtype
    )
    
    print("✅ Model loaded successfully!")
    
    # Show available datasets
    print("\nAvailable datasets:")
    kit.list_datasets()
    
    # Run evaluation
    dataset_name = "cure_bench_pharse_1"  # Update this to match your config
    print(f"\n🔬 Running evaluation on: {dataset_name}")
    
    results = kit.evaluate(dataset_name)
    
    # Save submission with metadata
    print("\n📊 Saving submission...")
    submission_path = kit.save_submission_with_metadata(
        results=[results],
        filename="unsloth_submission.csv",
        config_path="metadata_config_val.json"
    )
    
    print(f"\n✅ Evaluation completed!")
    print(f"📈 Accuracy: {results.accuracy:.2%} ({results.correct_predictions}/{results.total_examples})")
    print(f"💾 Submission saved to: {submission_path}")
    
    # Print some example predictions
    print(f"\n🔍 Sample predictions (first 3):")
    for i, pred in enumerate(results.predictions[:3]):
        print(f"  {i+1}. Choice: {pred.get('choice', 'N/A')}")
        print(f"     Answer: {pred.get('open_ended_answer', 'N/A')[:100]}...")
        print()

if __name__ == "__main__":
    main()
