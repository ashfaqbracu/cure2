#!/usr/bin/env python3
"""
Bio-Medical AI Competition - Unsloth Model Evaluation Script

This script demonstrates how to use the evaluation framework with Unsloth models.
Assumes you have already loaded your Unsloth model and tokenizer.

Usage:
    python run_unsloth.py --config metadata_config_unsloth.json
"""

import os
import argparse
from eval_framework import CompetitionKit, create_metadata_parser, load_and_merge_config

# Unsloth model loading (you need to modify this part based on your setup)
def load_unsloth_model():
    """
    Load Unsloth model - modify this function based on your model setup
    """
    try:
        from unsloth import FastModel
        import torch
        
        # Load your Unsloth model (replace with your specific model)
        model, tokenizer = FastModel.from_pretrained(
            model_name = "unsloth/medgemma-4b-it-bnb-4bit",
            dtype = None, # None for auto detection
            max_seq_length = 1024, # Choose any for long context!
            load_in_4bit = False,  # 4 bit quantization to reduce memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = "hf_...", # use one if using gated models
        )
        
        print(f"✅ Successfully loaded Unsloth model: unsloth/medgemma-4b-it-bnb-4bit")
        return model, tokenizer
        
    except ImportError as e:
        print(f"❌ Error: Failed to import Unsloth. Please install Unsloth first.")
        print(f"   pip install unsloth")
        raise e
    except Exception as e:
        print(f"❌ Error loading Unsloth model: {e}")
        raise e


def main():
    # Create argument parser
    parser = create_metadata_parser()
    args = parser.parse_args()
    
    # Load configuration from config file if provided and merge with args
    args = load_and_merge_config(args)
    
    # Extract values
    output_file = getattr(args, 'output_file', "unsloth_submission.csv") 
    dataset_name = getattr(args, 'dataset', 'cure_bench_phase_1')
    model_name = "unsloth/medgemma-4b-it-bnb-4bit"  # Your specific model name
    
    print("\n" + "="*60)
    print("🏥 CURE-Bench Competition - Unsloth Model Evaluation")
    print("="*60)
    
    # Load Unsloth model
    print("Loading Unsloth model...")
    model, tokenizer = load_unsloth_model()
    
    # Initialize the competition kit
    config_path = getattr(args, 'config', None)
    if not config_path:
        default_config = "metadata_config_unsloth.json"
        if os.path.exists(default_config):
            config_path = default_config
    
    kit = CompetitionKit(config_path=config_path)
    
    # Load model in the framework with Unsloth type
    print(f"Initializing framework with model: {model_name}")
    kit.load_model(
        model_name=model_name, 
        model_type="unsloth",
        model_instance=model,
        tokenizer_instance=tokenizer
    )
    
    # Show available datasets
    print("Available datasets:")
    kit.list_datasets()
    
    # Run evaluation
    print(f"Running evaluation on dataset: {dataset_name}")
    results = kit.evaluate(dataset_name)
    
    # Generate submission with metadata
    print("Generating submission with metadata...")
    submission_path = kit.save_submission_with_metadata(
        results=[results],
        filename=output_file,
        config_path=config_path,
        args=args
    )
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 Accuracy: {results.accuracy:.2%} ({results.correct_predictions}/{results.total_examples})")
    print(f"📄 Submission saved to: {submission_path}")
    
    # Show metadata summary
    final_metadata = kit.get_metadata(config_path, args)
    print("\n📋 Final metadata:")
    for key, value in final_metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
