#!/usr/bin/env python3
"""
Run core experiments - essential experiments for thesis.
Focused set of 9 experiments instead of 12 for faster completion.
"""

import subprocess
import time
import os
import sys
from datetime import datetime
from pathlib import Path


def run_experiment(config_name, experiment_name=None, wait_time=10):
    """Run single experiment using sn-reid framework."""
    if experiment_name is None:
        experiment_name = config_name
    
    config_path = f"benchmarks/baseline/configs/{config_name}.yaml"
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file {config_path} not found!")
        return False
    
    cmd = [
        sys.executable, 
        "benchmarks/baseline/main.py",
        "--config-file", config_path,
        "data.save_dir", f"log/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ]
    
    print(f"\n{'='*70}")
    print(f"CORE EXPERIMENT: {experiment_name}")
    print(f"CONFIG: {config_path}")
    print(f"TIME: {datetime.now().strftime('%H:%M:%S')}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            timeout=14400  # 4 hours timeout
        )
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"✅ COMPLETED ({elapsed/60:.1f} minutes)")
        
        # Extract key metrics from output
        lines = result.stdout.strip().split('\n')
        for line in lines[-20:]:
            if any(word in line.lower() for word in ['rank', 'map', 'accuracy', 'best']):
                print(f"  📊 {line}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (4 hours) - SKIPPING")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED (code: {e.returncode})")
        if e.stderr:
            print(f"Error details: {e.stderr[:500]}...")
        return False
    
    except Exception as e:
        print(f"💥 UNEXPECTED ERROR: {e}")
        return False
    
    finally:
        if wait_time > 0:
            print(f"⏸️  Waiting {wait_time} seconds...")
            time.sleep(wait_time)


def main():
    """Run core experiments for thesis (9 essential experiments)."""
    
    # Change to sn-reid-new directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    session_id = datetime.now().strftime('%m%d_%H%M')
    
    print("🎯 CORE EXPERIMENTS FOR THESIS")
    print("📋 Essential experiments covering all research aspects")
    print(f"🕐 Session: {session_id}")
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚙️  Settings: 60 epochs, 1% data, official sn-reid framework")
    print("⏱️  Target: ~48 hours TOTAL (4 hours per experiment)")
    
    # Core experiments - covers ALL thesis requirements
    experiments = [
        # 1. Baseline
        ("baseline_60epoch", "baseline_resnet50"),
        
        # 2-4. Architecture comparison (różne architektury ekstrakcji cech)
        ("arch_densenet121", "arch_densenet121"),
        ("arch_osnet", "arch_osnet"),
        ("optimized_experiment", "optimized_osnet"),
        
        # 5-6. Loss functions (różne funkcje straty - kontrastywna, syjamska)
        ("loss_pure_triplet", "loss_pure_triplet"),
        ("contrastive_loss", "contrastive_siamese"),
        
        # 7-9. Informatywne przykłady uczące (różne podejścia do budowania)
        ("sampling_many_instances", "sampling_8instances"),
        ("hard_mining_experiment", "hard_mining_osnet"),
        ("advanced_optimization", "advanced_informative_sampling"),
        
        # 10-11. Hyperparameter ablation
        ("ablation_margin_01", "margin_01"),
        ("ablation_margin_05", "margin_05"),
        
        # 12. Best combination
        ("best_combination", "best_combination"),
    ]
    
    print(f"📊 Core experiments: {len(experiments)}")
    print(f"🎯 Expected time: {len(experiments) * 4:.0f} hours")
    print("\n🔬 Research coverage (zgodnie z tematem pracy):")
    print("✓ Różne architektury ekstrakcji cech (ResNet50, DenseNet121, OSNet)")
    print("✓ Różne funkcje straty - kontrastywna/syjamska (Triplet, Contrastive)")
    print("✓ Różne podejścia do informatywnych przykładów (Random, Hard mining, Advanced)")
    print("✓ Hyperparameter sensitivity analysis")
    print("✓ Baseline vs Optimized approaches")
    print("✓ Best practice combination")
    
    # Start experiments
    completed = 0
    failed = 0
    start_time = time.time()
    
    for i, (config_name, experiment_name) in enumerate(experiments, 1):
        print(f"\n🚀 [{i}/{len(experiments)}] Starting: {experiment_name}")
        exp_start = time.time()
        
        if run_experiment(config_name, experiment_name):
            completed += 1
            exp_time = (time.time() - exp_start) / 60
            remaining = len(experiments) - i
            estimated_remaining = remaining * exp_time
            
            print(f"✅ SUCCESS: {experiment_name} done ({exp_time:.1f}min)")
            print(f"📈 Progress: {i}/{len(experiments)}")
            print(f"⏳ Est. remaining: {estimated_remaining:.1f}min ({estimated_remaining/60:.1f}h)")
        else:
            failed += 1
            print(f"❌ FAILED: {experiment_name} failed, continuing...")
    
    # Final summary
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*70)
    print("🎉 CORE EXPERIMENTS COMPLETED!")
    print(f"⏱️  Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"✅ Completed: {completed}/{len(experiments)}")
    print(f"❌ Failed: {failed}/{len(experiments)}")
    print(f"📊 Success rate: {completed/len(experiments)*100:.1f}%")
    
    print(f"\n📁 Results saved in: log/ directory")
    print("💡 Use tools/parse_test_res.py to analyze results")
    print("📝 Ready for thesis chapter: Experiments and Results")
    print("="*70)


if __name__ == "__main__":
    main()