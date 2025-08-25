#!/usr/bin/env python3
"""
Test script to verify the CompetitionKit can be initialized without config.
This demonstrates the fix for the exit() error in Colab/Jupyter environments.
"""

def test_framework_no_config():
    """Test framework initialization without config"""
    print("🧪 Testing framework initialization without config...")
    
    try:
        # Mock the torch import to avoid dependency issues in testing
        import sys
        from unittest.mock import MagicMock
        sys.modules['torch'] = MagicMock()
        
        from eval_framework import CompetitionKit
        
        # This should now work without throwing exit() error
        kit = CompetitionKit()
        print("✅ Framework initialized successfully without config!")
        
        # Test that we can manually set datasets
        test_dataset_config = {
            "dataset_name": "test_dataset",
            "dataset_path": "test.jsonl",
            "description": "Test dataset"
        }
        
        kit.datasets = {"test_dataset": test_dataset_config}
        print(f"✅ Manually set dataset: {list(kit.datasets.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_framework_with_config():
    """Test framework initialization with config"""
    print("\n🧪 Testing framework initialization with config...")
    
    try:
        import sys
        from unittest.mock import MagicMock
        sys.modules['torch'] = MagicMock()
        
        from eval_framework import CompetitionKit
        
        # Test with config file
        kit = CompetitionKit(config_path="metadata_config_unsloth.json")
        print(f"✅ Framework initialized with config! Datasets: {list(kit.datasets.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run tests"""
    print("🏥 Testing CompetitionKit Initialization")
    print("=" * 50)
    
    test1_pass = test_framework_no_config()
    test2_pass = test_framework_with_config()
    
    print(f"\n📋 Test Results:")
    print(f"   No Config Test:   {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"   With Config Test: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 All tests passed! The exit() error has been fixed.")
        print("📚 The framework now works properly in Jupyter/Colab environments.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
