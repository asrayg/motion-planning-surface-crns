"""
Generate a surface CRN manifest for motion planning.
The rule set is universal (independent of grid size); only the INIT_STATE varies.
"""

import sys
import random
from collections import deque
from itertools import product

POSITIONS = list(range(1, 10))

def pos(x, y):
    return 3 * (y % 3) + (x % 3) + 1

ADJACENT_PAIRS = [(p, q) for p in POSITIONS for q in POSITIONS]


def emit_colormap():
    lines = ["!START_COLORMAP"]
    lines.append("{empty} " + ", ".join(f"O{p}" for p in POSITIONS) + ": (240, 240, 240)")
    lines.append("{obstacle} X: (0, 0, 0)")
    lines.append("{goal} " + ", ".join(f"T{p}" for p in POSITIONS) + ": (255, 200, 0)")
    lines.append("{agent untagged} " + ", ".join(f"A{p}" for p in POSITIONS) + ": (220, 0, 0)")
    tagged = [f"At{t}_{p}" for t in POSITIONS for p in POSITIONS]
    lines.append("{agent tagged} " + ", ".join(tagged) + ": (150, 0, 150)")
    grad = [f"G{q}_{p}" for q in POSITIONS for p in POSITIONS]
    lines.append("{gradient} " + ", ".join(grad) + ": (100, 200, 100)")
    lines.append("{reached} " + ", ".join(f"R{p}" for p in POSITIONS) + ": (0, 0, 255)")
    lines.append("!END_COLORMAP")
    return "\n".join(lines)


def emit_rules():
    lines = ["!START_TRANSITION_RULES"]
    rate_fast = 10
    rate_slow = 1

    lines.append("# Family 1: Gradient initiation from goal")
    for p, q in ADJACENT_PAIRS:
        lines.append(f"T{p} + O{q} -> T{p} + G{p}_{q} ({rate_fast})")

    lines.append("# Family 2: Gradient propagation")
    for par in POSITIONS:
        for p, q in ADJACENT_PAIRS:
            lines.append(f"G{par}_{p} + O{q} -> G{par}_{p} + G{p}_{q} ({rate_fast})")

    lines.append("# Family 3a: Agent tagging")
    for par in POSITIONS:
        for pA, pG in ADJACENT_PAIRS:
            lines.append(f"A{pA} + G{par}_{pG} -> At{pG}_{pA} + G{par}_{pG} ({rate_slow})")

    lines.append("# Family 3b: Tagged agent move")
    for tag in POSITIONS:
        for par in POSITIONS:
            for pA, pG in ADJACENT_PAIRS:
                if tag == pG:
                    lines.append(f"At{tag}_{pA} + G{par}_{pG} -> O{pA} + At{par}_{pG} ({rate_slow})")

    lines.append("# Family 4: Termination (tagged)")
    for tag in POSITIONS:
        for pA, pT in ADJACENT_PAIRS:
            if tag == pT:
                lines.append(f"At{tag}_{pA} + T{pT} -> R{pA} + R{pT} ({rate_slow})")

    lines.append("# Family 4b: Direct termination (untagged)")
    for pA, pT in ADJACENT_PAIRS:
        lines.append(f"A{pA} + T{pT} -> R{pA} + R{pT} ({rate_slow})")

    lines.append("!END_TRANSITION_RULES")
    return "\n".join(lines)


def emit_init_3x3_simple():
    lines = ["!START_INIT_STATE"]
    rows = []
    for y in range(3):
        row = []
        for x in range(3):
            p = pos(x, y)
            if (x, y) == (0, 0):
                row.append(f"A{p}")
            elif (x, y) == (2, 2):
                row.append(f"T{p}")
            else:
                row.append(f"O{p}")
        rows.append(" ".join(row))
    lines.extend(rows)
    lines.append("!END_INIT_STATE")
    return "\n".join(lines)


def emit_init_5x5_with_obstacle():
    lines = ["!START_INIT_STATE"]
    rows = []
    for y in range(5):
        row = []
        for x in range(5):
            p = pos(x, y)
            if (x, y) == (0, 0):
                row.append(f"A{p}")
            elif (x, y) == (4, 4):
                row.append(f"T{p}")
            elif (x, y) == (2, 2):
                row.append("X")
            else:
                row.append(f"O{p}")
        rows.append(" ".join(row))
    lines.extend(rows)
    lines.append("!END_INIT_STATE")
    return "\n".join(lines)


def is_reachable(n, obstacles, start=(0, 0), goal=None):
    if goal is None:
        goal = (n - 1, n - 1)
    visited = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < n and 0 <= ny < n
                    and (nx, ny) not in obstacles
                    and (nx, ny) not in visited):
                visited.add((nx, ny))
                q.append((nx, ny))
    return False


def emit_init_NxN_random(n, obstacle_density=0.2, seed=42, max_retries=200):
    obstacles = None
    for attempt in range(max_retries):
        rng = random.Random(seed + attempt)
        candidate = set()
        for y in range(n):
            for x in range(n):
                if (x, y) in [(0, 0), (n - 1, n - 1)]:
                    continue
                if rng.random() < obstacle_density:
                    candidate.add((x, y))
        if is_reachable(n, candidate):
            obstacles = candidate
            break
    if obstacles is None:
        raise RuntimeError(
            f"Couldn't generate feasible {n}x{n} instance with density {obstacle_density}"
        )

    lines = ["!START_INIT_STATE"]
    rows = []
    for y in range(n):
        row = []
        for x in range(n):
            p = pos(x, y)
            if (x, y) == (0, 0):
                row.append(f"A{p}")
            elif (x, y) == (n - 1, n - 1):
                row.append(f"T{p}")
            elif (x, y) in obstacles:
                row.append("X")
            else:
                row.append(f"O{p}")
        rows.append(" ".join(row))
    lines.extend(rows)
    lines.append("!END_INIT_STATE")
    return "\n".join(lines)


HEADER = """# Motion planning on surface CRNs - auto-generated manifest
pixels_per_node    = 50
speedup_factor     = 5
max_duration       = 2000
node_display       = Color
rng_seed           = 42
"""


def parse_grid_spec(spec):
    """
    Grid spec format:
      "3x3"                    -> hardcoded 3x3 simple
      "5x5"                    -> hardcoded 5x5 with center obstacle
      "NxN_random"             -> NxN random, density 0.2, seed 42
      "NxN_random_D"           -> NxN random, density D, seed 42
      "NxN_random_D_S"         -> NxN random, density D, seed S
    """
    if spec == "3x3":
        return ("3x3", None)
    if spec == "5x5":
        return ("5x5", None)
    if "_random" in spec:
        parts = spec.split("_")
        size_part = parts[0]  # e.g. "15x15"
        n = int(size_part.split("x")[0])
        density = 0.2
        seed = 42
        if len(parts) >= 3:
            density = float(parts[2])
        if len(parts) >= 4:
            seed = int(parts[3])
        return ("random", {"n": n, "obstacle_density": density, "seed": seed})
    raise ValueError(f"Unknown grid spec: {spec}")


def main():
    grid_spec = sys.argv[1] if len(sys.argv) > 1 else "3x3"
    kind, params = parse_grid_spec(grid_spec)

    print(HEADER)
    print(emit_colormap())
    print()
    print(emit_rules())
    print()
    if kind == "3x3":
        print(emit_init_3x3_simple())
    elif kind == "5x5":
        print(emit_init_5x5_with_obstacle())
    elif kind == "random":
        print(emit_init_NxN_random(**params))


if __name__ == "__main__":
    main()