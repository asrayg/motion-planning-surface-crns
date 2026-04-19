"""
Generate a surface CRN manifest for motion planning.
The rule set is universal (independent of grid size); only the INIT_STATE varies.
"""

import sys
from itertools import product

# 3x3 position tiling: p(x,y) = 3*(y%3) + (x%3) + 1, values in {1..9}
POSITIONS = list(range(1, 10))

def pos(x, y):
    return 3 * (y % 3) + (x % 3) + 1

# Adjacency in the 3x3 tiling: which position labels can be neighbors?
# Two positions p, p' are "adjacency-compatible" if there exist sites (x,y) and (x',y')
# that are grid-neighbors with pos(x,y)=p, pos(x',y')=p'.
# For the full Z^2 tiling, every pair (p, p') with p, p' in 1..9 is adjacency-compatible
# somewhere (because the tiling repeats). So all 9x9=81 ordered pairs are valid.
# But within a specific 3x3 block, the adjacencies are constrained.
# For rule generality, we allow all 81 ordered pairs.
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
    rate_fast = 10  # gradient
    rate_slow = 1   # agent

    # Family 1: Gradient init from goal
    # T_p + O_p' -> T_p + G(p)_p'
    lines.append("# Family 1: Gradient initiation from goal")
    for p, q in ADJACENT_PAIRS:
        lines.append(f"T{p} + O{q} -> T{p} + G{p}_{q} ({rate_fast})")

    # Family 2: Gradient propagation
    # G(par)_p + O_p' -> G(par)_p + G(p)_p'
    lines.append("# Family 2: Gradient propagation")
    for par in POSITIONS:
        for p, q in ADJACENT_PAIRS:
            lines.append(f"G{par}_{p} + O{q} -> G{par}_{p} + G{p}_{q} ({rate_fast})")

    # Family 3a: Agent tagging (untagged agent + gradient neighbor)
    # A_pA + G(par)_pG -> At(pG)_pA + G(par)_pG
    lines.append("# Family 3a: Agent tagging")
    for par in POSITIONS:
        for pA, pG in ADJACENT_PAIRS:
            lines.append(f"A{pA} + G{par}_{pG} -> At{pG}_{pA} + G{par}_{pG} ({rate_slow})")

    # Family 3b: Tagged agent move
    # At(tag)_pA + G(par)_pG -> O_pA + At(par)_pG   IF tag == pG
    # (agent only moves onto its tagged neighbor)
    lines.append("# Family 3b: Tagged agent move")
    for tag in POSITIONS:
        for par in POSITIONS:
            for pA, pG in ADJACENT_PAIRS:
                if tag == pG:
                    lines.append(f"At{tag}_{pA} + G{par}_{pG} -> O{pA} + At{par}_{pG} ({rate_slow})")

    # Family 4: Termination (tagged agent adjacent to goal, with correct tag)
    # At(tag)_pA + T_pT -> R_pA + R_pT   IF tag == pT
    lines.append("# Family 4: Termination")
    for tag in POSITIONS:
        for pA, pT in ADJACENT_PAIRS:
            if tag == pT:
                lines.append(f"At{tag}_{pA} + T{pT} -> R{pA} + R{pT} ({rate_slow})")

    # Also: untagged agent directly adjacent to goal should terminate too
    # A_pA + T_pT -> R_pA + R_pT
    lines.append("# Family 4b: Direct termination (untagged agent adjacent to goal)")
    for pA, pT in ADJACENT_PAIRS:
        lines.append(f"A{pA} + T{pT} -> R{pA} + R{pT} ({rate_slow})")

    lines.append("!END_TRANSITION_RULES")
    return "\n".join(lines)

def emit_init_3x3_simple():
    """3x3 grid, agent at (0,0), goal at (2,2), no obstacles."""
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
    """5x5 grid, agent at (0,0), goal at (4,4), obstacle at (2,2)."""
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

HEADER = """# Motion planning on surface CRNs - auto-generated manifest
pixels_per_node    = 50
speedup_factor     = 1
max_duration       = 500
node_display       = Color
rng_seed           = 42
"""


def main():
    grid = sys.argv[1] if len(sys.argv) > 1 else "3x3"
    print(HEADER)
    print(emit_colormap())
    print()
    print(emit_rules())
    print()
    if grid == "3x3":
        print(emit_init_3x3_simple())
    elif grid == "5x5":
        print(emit_init_5x5_with_obstacle())
    else:
        raise ValueError(f"Unknown grid: {grid}")

if __name__ == "__main__":
    main()