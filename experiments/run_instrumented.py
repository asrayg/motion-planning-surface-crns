"""
Instrumented runner for motion-planning surface CRNs.
Logs reactions, tracks agent position, reports timing, and validates termination.
"""

import sys
import argparse
from surface_crns.readers.manifest_readers import read_manifest
from surface_crns.simulators.queue_simulator import QueueSimulator
from surface_crns.models.grids import SquareGrid


def classify_species(species):
    if species.startswith("At"):
        return "tagged_agent"
    if species.startswith("A") and len(species) <= 3:
        return "untagged_agent"
    if species.startswith("G") and "_" in species:
        return "gradient"
    if species.startswith("T"):
        return "goal"
    if species.startswith("O"):
        return "empty"
    if species.startswith("R") and species != "Rate":
        return "reached"
    if species == "X":
        return "obstacle"
    return "unknown"


def iter_nodes(grid):
    for r in range(grid.y_size):
        for c in range(grid.x_size):
            yield r, c, grid.getnode(c, r)


def summarize_grid(grid):
    counts = {"empty": 0, "obstacle": 0, "goal": 0, "untagged_agent": 0,
              "tagged_agent": 0, "gradient": 0, "reached": 0, "unknown": 0}
    for r, c, node in iter_nodes(grid):
        counts[classify_species(node.state)] += 1
    return counts


def find_species_sites(grid, predicate):
    return [(r, c, node.state) for r, c, node in iter_nodes(grid)
            if predicate(node.state)]


def run_instrumented(manifest_path, max_reactions=100000, verbose=True,
                     agent_trace=True, reaction_trace=False):
    options = read_manifest(manifest_path)
    init_state = options['init_state']
    transition_rules = options['transition_rules']
    max_duration = float(options.get('max_duration', 1000))
    rng_seed = options.get('rng_seed', None)
    if rng_seed is not None:
        try:
            rng_seed = int(rng_seed)
        except (ValueError, TypeError):
            rng_seed = None

    grid = SquareGrid(len(init_state[0]), len(init_state))
    grid.set_global_state(init_state)

    if verbose:
        print(f"=== Loaded manifest: {manifest_path}")
        print(f"Grid size: {grid.x_size} x {grid.y_size}")
        print(f"Number of rules: {len(transition_rules)}")
        print(f"Max duration: {max_duration}, RNG seed: {rng_seed}")
        print(f"\nInitial state counts:")
        for k, v in summarize_grid(grid).items():
            if v > 0:
                print(f"  {k}: {v}")

    sim = QueueSimulator(
        surface=grid,
        transition_rules=transition_rules,
        seed=rng_seed,
        simulation_duration=max_duration,
    )
    sim.initialize_reactions()

    def is_agent(s):
        return classify_species(s) in ("untagged_agent", "tagged_agent")
    def is_reached(s):
        return classify_species(s) == "reached"

    initial_agents = find_species_sites(grid, is_agent)
    if verbose:
        print(f"\nInitial agents: {initial_agents}")

    reaction_count = 0
    phase_log = {
        'first_gradient_time': None,
        'first_tagging_time': None,
        'first_agent_move_time': None,
        'reached_time': None,
        'agent_moves': 0,
        'gradient_reactions': 0,
        'tagging_reactions': 0,
        'termination_reactions': 0,
    }

    agent_path = []
    if initial_agents:
        r, c, s = initial_agents[0]
        agent_path.append((0.0, r, c, s))

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
        rule = event.rule
        reactants = list(rule.inputs)
        products = list(rule.outputs)
        reactant_cats = [classify_species(x) for x in reactants]
        product_cats = [classify_species(x) for x in products]

        # Classify reaction type
        if "reached" in product_cats:
            phase_log['termination_reactions'] += 1
            phase_log['reached_time'] = t
        elif "tagged_agent" in product_cats and "untagged_agent" in reactant_cats:
            phase_log['tagging_reactions'] += 1
            if phase_log['first_tagging_time'] is None:
                phase_log['first_tagging_time'] = t
        elif "tagged_agent" in product_cats and "tagged_agent" in reactant_cats:
            phase_log['agent_moves'] += 1
            if phase_log['first_agent_move_time'] is None:
                phase_log['first_agent_move_time'] = t
        elif "gradient" in product_cats and "gradient" not in reactant_cats:
            # Gradient emerging from O (either goal-initiated or propagated)
            phase_log['gradient_reactions'] += 1
            if phase_log['first_gradient_time'] is None:
                phase_log['first_gradient_time'] = t
        elif "gradient" in product_cats and "empty" in reactant_cats:
            phase_log['gradient_reactions'] += 1
            if phase_log['first_gradient_time'] is None:
                phase_log['first_gradient_time'] = t

        # Track agent
        is_agent_involved = ("agent" in " ".join(reactant_cats + product_cats)
                             or "reached" in product_cats)
        if agent_trace and is_agent_involved:
            agents_now = find_species_sites(grid, is_agent)
            reached_now = find_species_sites(grid, is_reached)
            if agents_now:
                r, c, s = agents_now[0]
                agent_path.append((t, r, c, s))
            elif reached_now and (not agent_path or not str(agent_path[-1][3]).startswith("REACHED")):
                r, c, s = reached_now[0]
                agent_path.append((t, r, c, f"REACHED({s})"))

        if reaction_trace:
            print(f"  t={t:7.3f} #{reaction_count:4d}: "
                  f"{' + '.join(reactants)} -> {' + '.join(products)}")

        if phase_log['reached_time'] is not None:
            # Let the termination reaction finalize, then stop
            break

    if verbose:
        print(f"\n=== Simulation complete")
        print(f"Total reactions fired: {reaction_count}")
        print(f"Final simulation time: {last_t:.4f}")
        print(f"\nPhase log:")
        for k, v in phase_log.items():
            print(f"  {k}: {v}")
        print(f"\nFinal state counts:")
        for k, v in summarize_grid(grid).items():
            if v > 0:
                print(f"  {k}: {v}")
        if agent_trace:
            print(f"\nAgent trajectory ({len(agent_path)} events):")
            for t, r, c, s in agent_path:
                print(f"  t={t:7.3f}  pos=(r={r},c={c})  state={s}")

    return {
        'reaction_count': reaction_count,
        'final_time': last_t,
        'phase_log': phase_log,
        'agent_path': agent_path,
        'final_state_counts': summarize_grid(grid),
        'reached': phase_log['reached_time'] is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--max-reactions", type=int, default=100000)
    ap.add_argument("--reactions", action="store_true",
                    help="Print every reaction as it fires")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    result = run_instrumented(
        args.manifest,
        max_reactions=args.max_reactions,
        verbose=not args.quiet,
        reaction_trace=args.reactions,
    )
    sys.exit(0 if result['reached'] else 1)


if __name__ == "__main__":
    main()