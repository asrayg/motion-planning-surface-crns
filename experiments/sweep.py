"""
Full empirical sweep for the paper.
Runs the motion-planning sCRN across grid sizes and obstacle densities,
collects timing and path-length data, and outputs a CSV.
"""

import sys
import os
import csv
import time
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from gen_manifest import (emit_colormap, emit_rules, emit_init_NxN_random,
                          HEADER, pos)
from run_instrumented import (classify_species, iter_nodes, summarize_grid)

from surface_crns.readers.manifest_readers import read_manifest
from surface_crns.simulators.queue_simulator import QueueSimulator
from surface_crns.models.grids import SquareGrid


def compute_shortest_path_distance(init_state):
    """BFS to find shortest path from agent to goal, avoiding obstacles."""
    n_rows = len(init_state)
    n_cols = len(init_state[0])
    start = None
    goal = None
    blocked = set()
    for r in range(n_rows):
        for c in range(n_cols):
            s = str(init_state[r][c])
            if s.startswith("A"):
                start = (r, c)
            elif s.startswith("T"):
                goal = (r, c)
            elif s == "X":
                blocked.add((r, c))
    if start is None or goal is None:
        return None, 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        if (r, c) == goal:
            n_traversable = n_rows * n_cols - len(blocked)
            return d, n_traversable
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n_rows and 0 <= nc < n_cols
                    and (nr, nc) not in blocked
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return None, n_rows * n_cols - len(blocked)


def generate_manifest_text(n, density, seed):
    return "\n".join([
        HEADER,
        emit_colormap(),
        "",
        emit_rules(),
        "",
        emit_init_NxN_random(n, obstacle_density=density, seed=seed),
    ])


def run_one_trial(n, density, seed, max_reactions=200000,
                  manifest_path="/tmp/sweep_manifest.txt"):
    """Run a single trial. Return dict with metrics, or None on failure."""
    try:
        manifest_text = generate_manifest_text(n, density, seed)
    except RuntimeError:
        return None  # infeasible instance

    Path(manifest_path).write_text(manifest_text)
    options = read_manifest(manifest_path)
    init_state = options["init_state"]
    transition_rules = options["transition_rules"]
    max_duration = float(options.get("max_duration", 2000))
    rng_seed = options.get("rng_seed", None)
    if rng_seed is not None:
        try:
            rng_seed = int(rng_seed)
        except (ValueError, TypeError):
            rng_seed = None

    d, n_traversable = compute_shortest_path_distance(init_state)
    if d is None:
        return None

    grid = SquareGrid(len(init_state[0]), len(init_state))
    grid.set_global_state(init_state)
    sim = QueueSimulator(
        surface=grid, transition_rules=transition_rules,
        seed=rng_seed, simulation_duration=max_duration,
    )
    sim.initialize_reactions()

    reaction_count = 0
    agent_moves = 0
    gradient_reactions = 0
    tagging_reactions = 0
    termination_reactions = 0
    first_gradient_time = None
    first_tagging_time = None
    first_agent_move_time = None
    reached_time = None
    last_t = 0.0

    while reaction_count < max_reactions:
        if sim.done():
            break
        event = sim.process_next_reaction()
        if event is None:
            break
        reaction_count += 1
        t = event.time
        last_t = t
        reactants = [classify_species(str(x)) for x in event.rule.inputs]
        products = [classify_species(str(x)) for x in event.rule.outputs]

        if "reached" in products:
            termination_reactions += 1
            reached_time = t
            break
        elif "tagged_agent" in products and "untagged_agent" in reactants:
            tagging_reactions += 1
            if first_tagging_time is None:
                first_tagging_time = t
        elif "tagged_agent" in products and "tagged_agent" in reactants:
            agent_moves += 1
            if first_agent_move_time is None:
                first_agent_move_time = t
        elif "gradient" in products:
            gradient_reactions += 1
            if first_gradient_time is None:
                first_gradient_time = t

    return {
        "n_param": n,
        "density": density,
        "seed": seed,
        "n_traversable": n_traversable,
        "d_shortest": d,
        "path_length": agent_moves + 1 if reached_time else None,
        "reactions": reaction_count,
        "gradient_reactions": gradient_reactions,
        "agent_moves": agent_moves,
        "tagging_reactions": tagging_reactions,
        "termination_reactions": termination_reactions,
        "final_time": last_t,
        "reached": reached_time is not None,
        "first_gradient_time": first_gradient_time,
        "first_tagging_time": first_tagging_time,
        "first_agent_move_time": first_agent_move_time,
        "reached_time": reached_time,
    }


def main():
    os.makedirs("data", exist_ok=True)
    output_path = "data/sweep.csv"

    grid_sizes = [5, 8, 10, 12, 15, 20]
    densities = [0.0, 0.1, 0.2, 0.3]
    trials_per_config = 30

    fieldnames = [
        "n_param", "density", "seed", "n_traversable", "d_shortest",
        "path_length", "reactions", "gradient_reactions", "agent_moves",
        "tagging_reactions", "termination_reactions", "final_time",
        "reached", "first_gradient_time", "first_tagging_time",
        "first_agent_move_time", "reached_time",
    ]

    t_start = time.time()
    total_configs = len(grid_sizes) * len(densities) * trials_per_config
    completed = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n in grid_sizes:
            for density in densities:
                for seed in range(trials_per_config):
                    result = run_one_trial(n, density, seed)
                    completed += 1
                    if result is None:
                        continue
                    writer.writerow(result)
                    f.flush()
                    if completed % 20 == 0:
                        elapsed = time.time() - t_start
                        rate = completed / elapsed
                        eta = (total_configs - completed) / rate
                        print(f"  [{completed}/{total_configs}] "
                              f"n={n}, rho={density:.1f}, seed={seed}: "
                              f"reactions={result['reactions']}, "
                              f"d={result['d_shortest']}, "
                              f"path={result['path_length']} "
                              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t_start
    print(f"\nDone. {completed} trials in {elapsed:.0f}s.")
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()