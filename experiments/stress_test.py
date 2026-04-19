"""
Stress test: run many random instances, check whether the agent always reaches the goal.
If any instance fails (agent gets stuck), we have a construction bug.
"""

import subprocess
import sys
import random
from pathlib import Path

# We regenerate the manifest with different seeds and check if the agent reaches.
# Uses the same generator.

sys.path.insert(0, str(Path(__file__).parent))
from gen_manifest import (emit_colormap, emit_rules, emit_init_NxN_random,
                          HEADER)

def generate_manifest(n, density, seed):
    return "\n".join([
        HEADER,
        emit_colormap(),
        "",
        emit_rules(),
        "",
        emit_init_NxN_random(n, obstacle_density=density, seed=seed),
    ])

def run_one(n, density, seed, manifest_path="/tmp/stress_manifest.txt",
            max_reactions=50000):
    manifest = generate_manifest(n, density, seed)
    Path(manifest_path).write_text(manifest)

    # Invoke run_instrumented.py in quiet mode
    result = subprocess.run(
        ["python", "experiments/run_instrumented.py", manifest_path, "--quiet",
         "--max-reactions", str(max_reactions)],
        capture_output=True, text=True
    )
    return result.returncode == 0  # 0 = reached, 1 = not reached

def main():
    trials = 50
    grid_sizes = [5, 8, 10, 12, 15]
    densities = [0.0, 0.1, 0.2, 0.3]

    print(f"{'n':>4} {'rho':>6} {'trials':>7} {'passed':>7} {'failed seeds':<30}")
    print("-" * 65)

    total_pass, total_fail = 0, 0
    all_failures = []

    for n in grid_sizes:
        for density in densities:
            passed, failed_seeds = 0, []
            for seed in range(trials):
                try:
                    if run_one(n, density, seed):
                        passed += 1
                    else:
                        failed_seeds.append(seed)
                except RuntimeError as e:
                    # generator couldn't find feasible instance
                    continue
            failed_str = ",".join(map(str, failed_seeds[:5]))
            if len(failed_seeds) > 5:
                failed_str += "..."
            print(f"{n:>4} {density:>6.2f} {passed+len(failed_seeds):>7} "
                  f"{passed:>7} {failed_str:<30}")
            total_pass += passed
            total_fail += len(failed_seeds)
            for s in failed_seeds:
                all_failures.append((n, density, s))

    print("-" * 65)
    print(f"Total: {total_pass} passed, {total_fail} failed")
    if all_failures:
        print("\nFailures to investigate:")
        for n, d, s in all_failures[:10]:
            print(f"  n={n}, density={d}, seed={s}")

if __name__ == "__main__":
    main()