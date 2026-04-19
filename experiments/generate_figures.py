"""
Generate figure snapshots for the paper.
Captures 4 states: initial, gradient-saturated (pre-tag), agent-mid-descent, reached.
"""
import sys
import os
from pathlib import Path
from surface_crns.readers.manifest_readers import read_manifest
from surface_crns.simulators.queue_simulator import QueueSimulator
from surface_crns.models.grids import SquareGrid

sys.path.insert(0, str(Path(__file__).parent))
from run_instrumented import classify_species, iter_nodes, summarize_grid


COLORS = {
    "empty":          (240, 240, 240),
    "obstacle":       (0, 0, 0),
    "goal":           (255, 200, 0),
    "untagged_agent": (220, 0, 0),
    "tagged_agent":   (150, 0, 150),
    "gradient":       (100, 200, 100),
    "reached":        (0, 0, 255),
    "unknown":        (255, 255, 255),
}


def grid_snapshot(grid):
    n_rows = grid.y_size
    n_cols = grid.x_size
    pixels = [[COLORS["unknown"]] * n_cols for _ in range(n_rows)]
    for r, c, node in iter_nodes(grid):
        cat = classify_species(str(node.state))
        pixels[r][c] = COLORS[cat]
    return pixels


def save_png(pixels, path, cell_size=40, border=2):
    from PIL import Image, ImageDraw
    n_rows = len(pixels)
    n_cols = len(pixels[0])
    W = n_cols * cell_size
    H = n_rows * cell_size
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * cell_size + border
            y0 = r * cell_size + border
            x1 = (c + 1) * cell_size - border
            y1 = (r + 1) * cell_size - border
            draw.rectangle([x0, y0, x1, y1], fill=pixels[r][c])
    img.save(path)
    print(f"  Saved {path}")


def run_dry(manifest_path, max_reactions=50000):
    """Dry run: count total reactions and find first_tagging reaction number."""
    options = read_manifest(manifest_path)
    init_state = options["init_state"]
    transition_rules = options["transition_rules"]
    max_duration = float(options.get("max_duration", 1000))
    rng_seed = options.get("rng_seed", None)
    if rng_seed is not None:
        try:
            rng_seed = int(rng_seed)
        except (ValueError, TypeError):
            rng_seed = None

    grid = SquareGrid(len(init_state[0]), len(init_state))
    grid.set_global_state(init_state)
    sim = QueueSimulator(
        surface=grid, transition_rules=transition_rules,
        seed=rng_seed, simulation_duration=max_duration,
    )
    sim.initialize_reactions()

    total = 0
    tagging_at = None
    pre_tag_at = None
    while total < max_reactions:
        if sim.done():
            break
        ev = sim.process_next_reaction()
        if ev is None:
            break
        total += 1
        reactants = [classify_species(str(x)) for x in ev.rule.inputs]
        products = [classify_species(str(x)) for x in ev.rule.outputs]
        if (tagging_at is None and "tagged_agent" in products
                and "untagged_agent" in reactants):
            tagging_at = total
            pre_tag_at = total - 1  # frame just before tagging
        if "reached" in products:
            break
    return total, tagging_at, pre_tag_at


def generate_figure_snapshots(manifest_path, out_dir, prefix, max_reactions=50000):
    total, tagging_at, pre_tag_at = run_dry(manifest_path, max_reactions)
    print(f"  Dry run: total={total}, tagging_at={tagging_at}")
    if tagging_at is None:
        print(f"  ERROR: agent never tagged. Skipping {prefix}.")
        return
    midway_target = tagging_at + (total - tagging_at) // 2
    print(f"  Snapshot targets: t0=0, pre-tag=~{pre_tag_at}, "
          f"midway={midway_target}, reached={total}")

    # Now actually run and save snapshots
    options = read_manifest(manifest_path)
    init_state = options["init_state"]
    transition_rules = options["transition_rules"]
    max_duration = float(options.get("max_duration", 1000))
    rng_seed = options.get("rng_seed", None)
    if rng_seed is not None:
        try:
            rng_seed = int(rng_seed)
        except (ValueError, TypeError):
            rng_seed = None

    grid = SquareGrid(len(init_state[0]), len(init_state))
    grid.set_global_state(init_state)

    os.makedirs(out_dir, exist_ok=True)
    pixels = grid_snapshot(grid)
    save_png(pixels, f"{out_dir}/{prefix}_t0_initial.png")

    sim = QueueSimulator(
        surface=grid, transition_rules=transition_rules,
        seed=rng_seed, simulation_duration=max_duration,
    )
    sim.initialize_reactions()

    reaction_count = 0
    saved_pretag = False
    saved_tagged = False
    saved_midway = False

    while reaction_count < max_reactions:
        if sim.done():
            break

        # Capture pre-tag BEFORE next reaction fires, if next would be tagging
        if (not saved_pretag and reaction_count + 1 == tagging_at):
            pixels = grid_snapshot(grid)
            save_png(pixels, f"{out_dir}/{prefix}_t1_gradient_saturated.png")
            saved_pretag = True

        event = sim.process_next_reaction()
        if event is None:
            break
        reaction_count += 1

        reactants = [classify_species(str(x)) for x in event.rule.inputs]
        products = [classify_species(str(x)) for x in event.rule.outputs]

        # Capture midway
        if not saved_midway and reaction_count >= midway_target:
            pixels = grid_snapshot(grid)
            save_png(pixels, f"{out_dir}/{prefix}_t2_descending.png")
            saved_midway = True

        # Capture reached
        if "reached" in products:
            pixels = grid_snapshot(grid)
            save_png(pixels, f"{out_dir}/{prefix}_t3_reached.png")
            break


def main():
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    from subprocess import run
    configs = [
        ("5x5", "motion_5x5.txt", "fig_5x5"),
        ("10x10_random", "motion_10x10_fig.txt", "fig_10x10"),
        ("15x15_random_0.2", "motion_15x15_fig.txt", "fig_15x15"),
    ]
    for spec, manifest_name, prefix in configs:
        manifest_path = f"manifests/{manifest_name}"
        run(["python", "experiments/gen_manifest.py", spec],
            stdout=open(manifest_path, "w"), check=True)
        print(f"\n=== {prefix} ({spec}) ===")
        generate_figure_snapshots(manifest_path, out_dir, prefix)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()