"""
Example: Using CureBench Evaluation Framework with Unsloth in Google Colab

This notebook shows how to evaluate Unsloth models on the CureBench dataset.
Perfect for Google Colab environment.

Steps:
1. Install dependencies
2. Load your Unsloth model
3. Run evaluation
4. Generate submission file
"""

# ================================================================
# 1. INSTALL DEPENDENCIES (Run this in Google Colab)
# ================================================================
"""
!pip install unsloth
# Install other required packages if needed
!pip install transformers torch tqdm pandas
"""

# ================================================================
# 2. LOAD UNSLOTH MODEL
# ================================================================
from unsloth import FastModel
import torch

# Load your Unsloth model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/medgemma-4b-it-bnb-4bit",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

print("✅ Unsloth model loaded successfully!")

# ================================================================
# 3. SETUP EVALUATION FRAMEWORK
# ================================================================
from eval_framework import CompetitionKit

# Create metadata for your submission
metadata = {
    "model_name": "unsloth/medgemma-4b-it-bnb-4bit",
    "model_type": "UnslothModel", 
    "track": "internal_reasoning",
    "base_model_type": "OpenWeighted",
    "base_model_name": "medgemma-4b-it",
    "dataset": "cure_bench_phase_1",
    "additional_info": "Unsloth optimized medical model"
}

# Create dataset configuration
dataset_config = {
    "dataset": {
        "dataset_name": "cure_bench_phase_1",
        "dataset_path": "curebench_testset_phase1.jsonl",  # Update path as needed
        "description": "CureBench 2025 Phase 1 test questions"
    },
    "output_dir": "results"
}

# Initialize competition kit
kit = CompetitionKit()
kit.datasets = {"cure_bench_phase_1": dataset_config["dataset"]}

# Load the model in the framework
kit.load_model(
    model_name="unsloth/medgemma-4b-it-bnb-4bit",
    model_type="unsloth",
    model_instance=model,
    tokenizer_instance=tokenizer
)

print("✅ Evaluation framework initialized!")

# ================================================================
# 4. RUN EVALUATION
# ================================================================
print("Running evaluation...")
results = kit.evaluate("cure_bench_phase_1")

print(f"✅ Evaluation completed!")
print(f"📊 Accuracy: {results.accuracy:.2%}")
print(f"📊 Correct: {results.correct_predictions}/{results.total_examples}")

# ================================================================
# 5. GENERATE SUBMISSION
# ================================================================
print("Generating submission file...")
submission_path = kit.save_submission_with_metadata(
    results=[results],
    metadata=metadata,
    filename="unsloth_submission.csv"
)

print(f"✅ Submission saved to: {submission_path}")
print("📄 Ready for competition submission!")

# ================================================================
# 6. DOWNLOAD RESULTS (In Colab)
# ================================================================
"""
# Download the submission file in Google Colab
from google.colab import files
files.download(submission_path)
"""
