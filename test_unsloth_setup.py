#!/usr/bin/env python3
"""
Test script to verify Unsloth integration with CureBench evaluation framework.

This script tests the framework without requiring actual model loading,
useful for testing the integration before running full evaluation.
"""

def test_framework_setup():
    """Test if the evaluation framework can handle Unsloth models"""
    print("🧪 Testing CureBench framework with Unsloth integration...")
    
    try:
        from eval_framework import CompetitionKit, UnslothModel
        print("✅ Successfully imported evaluation framework with Unsloth support")
    except ImportError as e:
        print(f"❌ Failed to import framework: {e}")
        return False
    
    try:
        # Test framework initialization
        kit = CompetitionKit()
        print("✅ Framework initialization successful")
    except Exception as e:
        print(f"❌ Framework initialization failed: {e}")
        return False
    
    print("✅ All tests passed! Framework is ready for Unsloth models.")
    return True

def test_config_loading():
    """Test configuration file loading"""
    print("\n🧪 Testing configuration loading...")
    
    import os
    config_file = "metadata_config_unsloth.json"
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        return False
    
    try:
        from eval_framework import CompetitionKit
        kit = CompetitionKit(config_path=config_file)
        print("✅ Configuration loaded successfully")
        
        # Test dataset configuration
        if hasattr(kit, 'datasets') and kit.datasets:
            print(f"✅ Found {len(kit.datasets)} dataset(s) configured")
            for name, config in kit.datasets.items():
                print(f"   - {name}: {config.get('description', 'No description')}")
        else:
            print("⚠️ No datasets found in configuration")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def show_usage_example():
    """Show example usage code"""
    print("\n📖 Example usage with Unsloth:")
    print("""
# 1. Load your Unsloth model (in Colab or local environment)
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/medgemma-4b-it-bnb-4bit",
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=True,
    full_finetuning=False,
)

# 2. Initialize evaluation framework
from eval_framework import CompetitionKit
kit = CompetitionKit(config_path="metadata_config_unsloth.json")

# 3. Load model in framework
kit.load_model(
    model_name="unsloth/medgemma-4b-it-bnb-4bit",
    model_type="unsloth",
    model_instance=model,
    tokenizer_instance=tokenizer
)

# 4. Run evaluation
results = kit.evaluate("cure_bench_phase_1")

# 5. Generate submission
submission_path = kit.save_submission_with_metadata(results=[results])
""")

def main():
    """Run all tests"""
    print("🏥 CureBench Unsloth Integration Test")
    print("=" * 50)
    
    # Test framework setup
    framework_ok = test_framework_setup()
    
    # Test configuration loading
    config_ok = test_config_loading()
    
    # Show usage example
    show_usage_example()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"   Framework Setup: {'✅ PASS' if framework_ok else '❌ FAIL'}")
    print(f"   Configuration:   {'✅ PASS' if config_ok else '❌ FAIL'}")
    
    if framework_ok and config_ok:
        print("\n🎉 All tests passed! You're ready to use Unsloth models with CureBench.")
        print("📚 Next steps:")
        print("   1. Load your Unsloth model using FastModel.from_pretrained()")
        print("   2. Use the evaluation framework as shown in the example")
        print("   3. Or run the Jupyter notebook: CureBench_Unsloth_Evaluation.ipynb")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
