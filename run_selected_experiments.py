#!/usr/bin/env python3
"""
Run selected experiments based on command line parameters.
Flexible experiment runner allowing user to choose specific experiments.
"""

import argparse
import subprocess
import time
import os
import sys
from datetime import datetime
from pathlib import Path


# Available experiments with descriptions
AVAILABLE_EXPERIMENTS = {
    # Baseline experiments
    'baseline': {
        'config': 'baseline_60epoch',
        'name': 'baseline_resnet50',
        'description': 'Baseline ResNet50 with standard settings'
    },
    'enhanced_baseline': {
        'config': 'enhanced_baseline',
        'name': 'enhanced_resnet50',
        'description': 'Enhanced baseline with improved augmentations'
    },
    
    # Architecture experiments
    'densenet': {
        'config': 'arch_densenet121',
        'name': 'arch_densenet121',
        'description': 'DenseNet121 architecture comparison'
    },
    'osnet': {
        'config': 'arch_osnet',
        'name': 'arch_osnet',
        'description': 'OSNet architecture comparison'
    },
    'optimized': {
        'config': 'optimized_experiment',
        'name': 'optimized_osnet',
        'description': 'Optimized OSNet with enhanced settings'
    },
    
    # Loss function experiments
    'triplet': {
        'config': 'loss_pure_triplet',
        'name': 'loss_pure_triplet',
        'description': 'Pure triplet loss without cross-entropy'
    },
    'contrastive': {
        'config': 'contrastive_loss',
        'name': 'contrastive_siamese',
        'description': 'Contrastive loss with siamese network'
    },
    
    # Sampling experiments
    'sampling': {
        'config': 'sampling_many_instances',
        'name': 'sampling_8instances',
        'description': 'Many instances sampling strategy (8 per identity)'
    },
    'hard_mining': {
        'config': 'hard_mining_experiment',
        'name': 'hard_mining_osnet',
        'description': 'Hard negative mining strategy'
    },
    'advanced': {
        'config': 'advanced_optimization',
        'name': 'advanced_osnet',
        'description': 'Advanced optimization techniques'
    },
    
    # Ablation studies
    'margin_01': {
        'config': 'ablation_margin_01',
        'name': 'margin_01',
        'description': 'Triplet margin ablation (0.1)'
    },
    'margin_05': {
        'config': 'ablation_margin_05',
        'name': 'margin_05',
        'description': 'Triplet margin ablation (0.5)'
    },
    
    # Special experiments
    'best': {
        'config': 'best_combination',
        'name': 'best_combination',
        'description': 'Best combination of all techniques'
    },
    'cosine': {
        'config': 'cosine_distance',
        'name': 'cosine_distance',
        'description': 'Cosine distance metric experiment'
    },
    'warmup': {
        'config': 'warmup_experiment',
        'name': 'warmup_experiment',
        'description': 'Learning rate warmup experiment'
    }
}

# Predefined experiment groups
EXPERIMENT_GROUPS = {
    'core': [
        'baseline', 'densenet', 'osnet', 'optimized',
        'triplet', 'contrastive', 'sampling', 'hard_mining',
        'margin_01', 'margin_05', 'best'
    ],
    'architecture': ['baseline', 'densenet', 'osnet', 'optimized'],
    'loss': ['baseline', 'triplet', 'contrastive'],
    'sampling': ['baseline', 'sampling', 'hard_mining', 'advanced'],
    'ablation': ['baseline', 'margin_01', 'margin_05'],
    'all': list(AVAILABLE_EXPERIMENTS.keys()),
    'quick': ['baseline', 'osnet', 'triplet'],
    'baseline_only': ['baseline'],
    'best_only': ['best']
}


def run_experiment(config_name, experiment_name, wait_time=10, timeout=14400):
    """Run single experiment using sn-reid framework."""
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
    print(f"EXPERIMENT: {experiment_name}")
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
            timeout=timeout
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
        print(f"⏰ TIMEOUT ({timeout/3600:.1f} hours) - SKIPPING")
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


def parse_experiments(experiment_args):
    """Parse experiment arguments and return list of experiments to run."""
    experiments_to_run = []
    
    for arg in experiment_args:
        if arg in EXPERIMENT_GROUPS:
            # It's a group
            group_experiments = EXPERIMENT_GROUPS[arg]
            experiments_to_run.extend(group_experiments)
            print(f"📋 Added group '{arg}': {group_experiments}")
        elif arg in AVAILABLE_EXPERIMENTS:
            # It's a single experiment
            experiments_to_run.append(arg)
            print(f"✅ Added experiment: {arg}")
        else:
            print(f"⚠️  Unknown experiment/group: {arg}")
            print("Available experiments:", list(AVAILABLE_EXPERIMENTS.keys()))
            print("Available groups:", list(EXPERIMENT_GROUPS.keys()))
    
    # Remove duplicates while preserving order
    unique_experiments = []
    for exp in experiments_to_run:
        if exp not in unique_experiments:
            unique_experiments.append(exp)
    
    return unique_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Run selected experiments for SoccerNet Re-ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available experiments:
{chr(10).join([f"  {name:15} - {info['description']}" for name, info in AVAILABLE_EXPERIMENTS.items()])}

Available groups:
  core          - Essential experiments for thesis ({len(EXPERIMENT_GROUPS['core'])} experiments)
  architecture  - Architecture comparison ({len(EXPERIMENT_GROUPS['architecture'])} experiments)
  loss          - Loss function comparison ({len(EXPERIMENT_GROUPS['loss'])} experiments)
  sampling      - Sampling strategy comparison ({len(EXPERIMENT_GROUPS['sampling'])} experiments)
  ablation      - Ablation studies ({len(EXPERIMENT_GROUPS['ablation'])} experiments)
  all           - All available experiments ({len(EXPERIMENT_GROUPS['all'])} experiments)
  quick         - Quick test with 3 experiments
  baseline_only - Only baseline experiment
  best_only     - Only best combination experiment

Examples:
  python run_selected_experiments.py baseline osnet triplet
  python run_selected_experiments.py core
  python run_selected_experiments.py architecture loss
  python run_selected_experiments.py --list
  python run_selected_experiments.py baseline --timeout 7200 --wait 30
        """
    )
    
    parser.add_argument(
        'experiments',
        nargs='*',
        help='Experiments or groups to run (see list below)'
    )
    
    parser.add_argument(
        '--list', '--show',
        action='store_true',
        help='Show available experiments and groups, then exit'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=14400,
        help='Timeout per experiment in seconds (default: 14400 = 4 hours)'
    )
    
    parser.add_argument(
        '--wait',
        type=int,
        default=10,
        help='Wait time between experiments in seconds (default: 10)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without actually running experiments'
    )
    
    args = parser.parse_args()
    
    # Show available experiments and exit
    if args.list:
        print("📋 AVAILABLE EXPERIMENTS:")
        for name, info in AVAILABLE_EXPERIMENTS.items():
            print(f"  {name:15} - {info['description']}")
        
        print(f"\n📁 AVAILABLE GROUPS:")
        for group, experiments in EXPERIMENT_GROUPS.items():
            print(f"  {group:15} - {len(experiments)} experiments: {', '.join(experiments[:3])}{'...' if len(experiments) > 3 else ''}")
        
        return
    
    # If no experiments specified, show help
    if not args.experiments:
        parser.print_help()
        return
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Parse experiments to run
    experiments_to_run = parse_experiments(args.experiments)
    
    if not experiments_to_run:
        print("❌ No valid experiments found!")
        return
    
    # Show experiment plan
    session_id = datetime.now().strftime('%m%d_%H%M')
    print(f"\n🎯 SELECTED EXPERIMENTS SESSION")
    print(f"🕐 Session: {session_id}")
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Timeout per experiment: {args.timeout/3600:.1f} hours")
    print(f"⏸️  Wait between experiments: {args.wait} seconds")
    
    print(f"\n📊 Experiments to run ({len(experiments_to_run)}):")
    total_estimated_time = 0
    for i, exp_key in enumerate(experiments_to_run, 1):
        exp_info = AVAILABLE_EXPERIMENTS[exp_key]
        estimated_hours = args.timeout / 3600
        total_estimated_time += estimated_hours
        print(f"  {i:2d}. {exp_key:15} - {exp_info['description']}")
    
    print(f"\n⏳ Estimated total time: {total_estimated_time:.1f} hours")
    print(f"🎯 Expected completion: ~{total_estimated_time} hours from now")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No experiments will be executed")
        print("Remove --dry-run flag to actually run the experiments")
        return
    
    # Confirm execution
    try:
        response = input(f"\n🚀 Ready to run {len(experiments_to_run)} experiments? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Aborted by user")
            return
    except KeyboardInterrupt:
        print("\n❌ Aborted by user")
        return
    
    # Run experiments
    completed = 0
    failed = 0
    start_time = time.time()
    
    for i, exp_key in enumerate(experiments_to_run, 1):
        exp_info = AVAILABLE_EXPERIMENTS[exp_key]
        print(f"\n🚀 [{i}/{len(experiments_to_run)}] Starting: {exp_key}")
        print(f"📝 Description: {exp_info['description']}")
        
        exp_start = time.time()
        
        if run_experiment(exp_info['config'], exp_info['name'], args.wait, args.timeout):
            completed += 1
            exp_time = (time.time() - exp_start) / 60
            remaining = len(experiments_to_run) - i
            estimated_remaining = remaining * exp_time
            
            print(f"✅ SUCCESS: {exp_key} completed ({exp_time:.1f}min)")
            print(f"📈 Progress: {i}/{len(experiments_to_run)}")
            print(f"⏳ Est. remaining: {estimated_remaining:.1f}min ({estimated_remaining/60:.1f}h)")
        else:
            failed += 1
            print(f"❌ FAILED: {exp_key} failed, continuing...")
    
    # Final summary
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*70)
    print("🎉 SELECTED EXPERIMENTS COMPLETED!")
    print(f"⏱️  Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"✅ Completed: {completed}/{len(experiments_to_run)}")
    print(f"❌ Failed: {failed}/{len(experiments_to_run)}")
    print(f"📊 Success rate: {completed/len(experiments_to_run)*100:.1f}%")
    
    print(f"\n📁 Results saved in: log/ directory")
    print("💡 Use tools/parse_test_res.py to analyze results")
    print("="*70)


if __name__ == "__main__":
    main()