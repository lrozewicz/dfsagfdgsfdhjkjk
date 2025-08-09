#!/usr/bin/env python3
"""
Quick test script - runs baseline experiment with minimal settings for testing.
Useful for verifying setup before running full experiments.
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def main():
    """Run quick test experiment to verify setup."""
    
    # Change to sn-reid-new directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🧪 SN-REID QUICK TEST")
    print("⚡ Fast experiment to verify setup (5 epochs, 0.1% data)")
    print(f"🕐 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create quick test config
    quick_config = """# Quick test configuration
model:
  name: 'resnet50_fc512'

data:
  type: 'image'
  root: 'datasets'
  height: 256
  width: 128
  workers: 2
  sources: ['soccernetv3']
  targets: ['soccernetv3']

soccernetv3:
  training_subset: 0.001  # Only 0.1% of data for quick test

sampler:
  train_sampler: RandomIdentitySampler
  train_sampler_t: RandomIdentitySampler
  num_instances: 2

loss:
  name: 'triplet'
  softmax:
    label_smooth: True
  triplet:
    margin: 0.3
    weight_t: 0.5
    weight_x: 0.5

train:
  batch_size: 16  # Small batch for quick test
  print_freq: 5
  max_epoch: 5    # Only 5 epochs for quick test
  lr: 0.0003
  weight_decay: 5e-4

test:
  ranks: [1, 5]
  export_ranking_results: False
  eval_freq: 2  # Evaluate every 2 epochs
"""
    
    # Save quick config
    config_path = "benchmarks/baseline/configs/quick_test.yaml"
    with open(config_path, 'w') as f:
        f.write(quick_config)
    
    # Run quick test
    cmd = [
        sys.executable, 
        "benchmarks/baseline/main.py",
        "--config-file", config_path,
        "data.save_dir", f"log/quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, timeout=1200)  # 20 min timeout
        print("\n" + "=" * 50)
        print("✅ QUICK TEST COMPLETED SUCCESSFULLY!")
        print("✅ Setup is working correctly")
        print("✅ Ready to run full experiments with run_all_experiments.py")
        
    except subprocess.TimeoutExpired:
        print("\n❌ TIMEOUT: Test took too long (>20min)")
        print("💡 Check GPU availability and dataset download")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: Test failed with code {e.returncode}")
        print("💡 Check error messages above for troubleshooting")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        
    finally:
        # Clean up quick test config
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    main()