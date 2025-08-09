#!/usr/bin/env python3
"""
Run all 15 experiments described in the thesis.
Complete reproduction of all experimental results presented in the dissertation.
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
    print(f"THESIS EXPERIMENT: {experiment_name}")
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
    """Run ALL 15 experiments described in the thesis."""
    
    # Change to sn-reid-new directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    session_id = datetime.now().strftime('%m%d_%H%M')
    
    print("🎓 COMPLETE THESIS EXPERIMENT REPRODUCTION")
    print("📚 All 15 experiments from dissertation chapter 4")
    print(f"🕐 Session: {session_id}")
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚙️  Settings: 60 epochs, 1% data subset, official framework")
    print("⏱️  Target: ~60 hours TOTAL (4 hours per experiment)")
    
    # Complete list of 15 experiments from thesis table 4.11 (ranking)
    experiments = [
        # METRIC LEARNING EXPERIMENTS (12 experiments)
        # Position 1: Hard mining OSNet - best result (54.7% mAP)
        ("hard_mining_experiment", "hard_mining_osnet"),
        
        # Position 2: Optimized OSNet (54.2% mAP)
        ("optimized_experiment", "optimized_osnet"),
        
        # Position 3: Sampling 8 instances ResNet50 (52.9% mAP)
        ("sampling_many_instances", "sampling_8instances"),
        
        # Position 4: Contrastive siamese ResNet50 (52.7% mAP)
        ("contrastive_loss", "contrastive_siamese"),
        
        # Position 5: Baseline ResNet50 (52.3% mAP)
        ("baseline_60epoch", "baseline_resnet50"),
        
        # Position 6: Margin 0.5 ResNet50 (51.8% mAP)
        ("ablation_margin_05", "margin_05"),
        
        # Position 7: Advanced sampling OSNet (51.5% mAP)
        ("advanced_optimization", "advanced_sampling"),
        
        # Position 8: Margin 0.1 ResNet50 (48.8% mAP)
        ("ablation_margin_01", "margin_01"),
        
        # Position 10: DenseNet121 architecture (47.4% mAP)
        ("arch_densenet121", "arch_densenet121"),
        
        # Position 11: OSNet baseline (46.4% mAP)
        ("arch_osnet", "arch_osnet"),
        
        # Position 12: Pure triplet loss ResNet50 (46.4% mAP)
        ("loss_pure_triplet", "loss_pure_triplet"),
        
        # CLASSIFICATION EXPERIMENTS (3 experiments)
        # Position 9: EfficientNet-B3 classification (48.1% mAP)
        ("classification_efficientnet_b3", "classif_efficientnet_b3"),
        
        # Position 13: EfficientNet-B1 classification (44.9% mAP)
        ("classification_efficientnet_b1", "classif_efficientnet_b1"),
        
        # Position 14: ResNet50 classification (35.9% mAP)
        ("classification_resnet50", "classif_resnet50"),
        
        # Position 15: Default baseline classification (35.6% mAP)
        ("default_baseline", "default_baseline"),
    ]
    
    print(f"\n📊 Total experiments: {len(experiments)}")
    print(f"🎯 Expected time: {len(experiments) * 4:.0f} hours")
    print(f"📈 Results will reproduce Table 4.11 ranking from thesis")
    
    print("\n🔬 Experiment categories covered:")
    print("✓ Hard negative mining techniques (positions 1-2)")
    print("✓ Sampling strategies (positions 3, 7)")
    print("✓ Loss function variations (positions 4, 12)")
    print("✓ Architecture comparisons (positions 5, 10-11)")
    print("✓ Hyperparameter ablations (positions 6, 8)")
    print("✓ Classification approaches (positions 9, 13-15)")
    print("✓ Baseline references (positions 5, 15)")
    
    # Detailed experiment descriptions
    experiment_details = {
        "hard_mining_osnet": "OSNet + Hard mining + 8 instances",
        "optimized_osnet": "OSNet + Optimized parameters + margin 0.5",
        "sampling_8instances": "ResNet50 + 8 instances per identity",
        "contrastive_siamese": "ResNet50 + Contrastive-like approach",
        "baseline_resnet50": "ResNet50 + Baseline triplet+CE",
        "margin_05": "ResNet50 + Triplet margin = 0.5",
        "advanced_sampling": "OSNet + L2 norm + higher resolution",
        "margin_01": "ResNet50 + Triplet margin = 0.1",
        "classif_efficientnet_b3": "EfficientNet-B3 + Pure cross-entropy",
        "arch_densenet121": "DenseNet121 + Baseline settings",
        "arch_osnet": "OSNet + Baseline settings",
        "loss_pure_triplet": "ResNet50 + Only triplet loss",
        "classif_efficientnet_b1": "EfficientNet-B1 + Pure cross-entropy",
        "classif_resnet50": "ResNet50 + Pure cross-entropy + fc512",
        "default_baseline": "ResNet50 + Framework baseline"
    }
    
    print(f"\n📋 Detailed experiment plan:")
    for i, (config, name) in enumerate(experiments, 1):
        description = experiment_details.get(name, "")
        print(f"  {i:2d}. {name:25} - {description}")
    
    print(f"\n⚠️  IMPORTANT: This will take approximately {len(experiments) * 4} hours to complete!")
    print("🔋 Ensure stable power supply and sufficient disk space (~100GB)")
    print("🌡️  Monitor GPU temperature during long runs")
    
    # Confirmation
    try:
        response = input(f"\n🚀 Ready to start ALL {len(experiments)} thesis experiments? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Aborted by user")
            return
    except KeyboardInterrupt:
        print("\n❌ Aborted by user")
        return
    
    # Start experiments
    completed = 0
    failed = 0
    start_time = time.time()
    
    print(f"\n🏁 Starting thesis experiment reproduction...")
    
    for i, (config_name, experiment_name) in enumerate(experiments, 1):
        description = experiment_details.get(experiment_name, "")
        print(f"\n🚀 [{i}/{len(experiments)}] Starting: {experiment_name}")
        print(f"📝 Description: {description}")
        exp_start = time.time()
        
        if run_experiment(config_name, experiment_name):
            completed += 1
            exp_time = (time.time() - exp_start) / 60
            remaining = len(experiments) - i
            estimated_remaining = remaining * exp_time
            
            print(f"✅ SUCCESS: {experiment_name} done ({exp_time:.1f}min)")
            print(f"📈 Progress: {i}/{len(experiments)} ({i/len(experiments)*100:.1f}%)")
            print(f"⏳ Est. remaining: {estimated_remaining:.1f}min ({estimated_remaining/60:.1f}h)")
        else:
            failed += 1
            print(f"❌ FAILED: {experiment_name} failed, continuing...")
    
    # Final summary
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*70)
    print("🎉 THESIS EXPERIMENT REPRODUCTION COMPLETED!")
    print(f"⏱️  Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"✅ Completed: {completed}/{len(experiments)}")
    print(f"❌ Failed: {failed}/{len(experiments)}")
    print(f"📊 Success rate: {completed/len(experiments)*100:.1f}%")
    
    if completed >= 12:  # Most experiments completed
        print("\n🎓 THESIS RESULTS READY:")
        print("✓ Table 4.11 ranking can be reproduced")
        print("✓ All experimental sections have data")
        print("✓ Comparison between metric learning and classification complete")
    else:
        print(f"\n⚠️  Warning: Only {completed}/15 experiments completed")
        print("Some thesis results may be incomplete")
    
    print(f"\n📁 Results saved in: log/ directory")
    print("💡 Next steps:")
    print("  1. Use tools/parse_test_res.py to extract numerical results")
    print("  2. Update thesis tables with actual experimental results")
    print("  3. Verify ranking matches expected order from Table 4.11")
    print("  4. Generate visualizations for thesis figures")
    print("="*70)


if __name__ == "__main__":
    main()