"""
ElasticNet Attack Pipeline - Complete Workflow

Run all three stages:
    python3 main.py

Or run individually:
    python3 train_model.py     # Stage 1: Train/load model
    python3 run_attack.py      # Stage 2: Generate adversarial examples
    python3 generate_plots.py  # Stage 3: Create visualizations
"""

import subprocess
import sys


def run_stage(script_name: str, stage_number: int, stage_name: str) -> bool:
    """Execute a pipeline stage script."""
    print(f"\n{'='*70}")
    print(f"STAGE {stage_number}: {stage_name}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
        )
        print(f"\n✓ Stage {stage_number} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Stage {stage_number} failed with exit code {e.returncode}")
        return False


def main() -> None:
    """Run the complete ElasticNet attack pipeline."""
    print("\n" + "="*70)
    print("ElasticNet Adversarial Attack Pipeline")
    print("="*70)

    stages = [
        ("train_model.py", 1, "Train/Load MNIST Model"),
        ("run_attack.py", 2, "Generate Adversarial Examples"),
        ("generate_plots.py", 3, "Create Visualizations"),
    ]

    failed_stages = []

    for script, stage_num, stage_name in stages:
        success = run_stage(script, stage_num, stage_name)
        if not success:
            failed_stages.append((stage_num, stage_name))

    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")

    if not failed_stages:
        print("✓ All stages completed successfully!")
        print("\nOutput files generated in output/:")
        print("  - mnist_target.pth        (trained model)")
        print("  - attack_result.pt        (attack results)")
        print("  - ead_attack_process.png  (visualization)")
        print("  - ead_distortion_analysis.png")
        print("  - ead_success_analysis.png")
        print("  - ead_sparsity_analysis.png")
    else:
        print(f"✗ {len(failed_stages)} stage(s) failed:")
        for stage_num, stage_name in failed_stages:
            print(f"  - Stage {stage_num}: {stage_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()