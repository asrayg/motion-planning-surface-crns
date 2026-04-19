"""Discover the actual API of the installed surface_crns package."""
import surface_crns
from surface_crns.models.grids import SquareGrid
from surface_crns.simulators.queue_simulator import QueueSimulator
from surface_crns.readers.manifest_readers import read_manifest

print("=" * 60)
print("surface_crns version:", getattr(surface_crns, '__version__', 'unknown'))
print("=" * 60)

print("\n--- SquareGrid public methods ---")
for attr in sorted(dir(SquareGrid)):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\n--- QueueSimulator public methods ---")
for attr in sorted(dir(QueueSimulator)):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Load the manifest and see what read_manifest returns
import sys
if len(sys.argv) > 1:
    result = read_manifest(sys.argv[1])
    print(f"\n--- read_manifest returned type: {type(result)} ---")
    if isinstance(result, tuple):
        print(f"  tuple of length {len(result)}")
        for i, item in enumerate(result):
            print(f"  [{i}] {type(item).__name__}")
            if hasattr(item, '__len__'):
                try:
                    print(f"      len = {len(item)}")
                except:
                    pass
    elif isinstance(result, dict):
        print(f"  dict with keys: {list(result.keys())}")

    # Build a grid and introspect a node
    init_state = None
    transition_rules = None
    if isinstance(result, dict):
        init_state = result.get('init_state')
        transition_rules = result.get('transition_rules')
    elif isinstance(result, tuple):
        # guess positions
        for item in result:
            if isinstance(item, list) and item and isinstance(item[0], list):
                init_state = item
            elif isinstance(item, list) and item and hasattr(item[0], 'inputs'):
                transition_rules = item
            elif isinstance(item, list) and item and hasattr(item[0], 'reactants'):
                transition_rules = item

    if init_state is not None:
        print(f"\n--- init_state: {len(init_state)} rows x {len(init_state[0])} cols ---")
        grid = SquareGrid(len(init_state), len(init_state[0]))
        print(f"\n--- SquareGrid instance attributes ---")
        for attr in sorted(dir(grid)):
            if not attr.startswith('_') and not callable(getattr(grid, attr)):
                try:
                    val = getattr(grid, attr)
                    print(f"  {attr} = {val!r}"[:100])
                except:
                    pass

        # Set state
        if hasattr(grid, 'set_global_state'):
            grid.set_global_state(init_state)
        # Try to fetch a node
        if hasattr(grid, 'getnode'):
            node = grid.getnode(0, 0)
            print(f"\n--- grid.getnode(0, 0) returned type: {type(node).__name__} ---")
            for attr in sorted(dir(node)):
                if not attr.startswith('_'):
                    print(f"  {attr}")

    if transition_rules:
        print(f"\n--- transition_rules: {len(transition_rules)} rules ---")
        rule = transition_rules[0]
        print(f"Rule 0 type: {type(rule).__name__}")
        for attr in sorted(dir(rule)):
            if not attr.startswith('_'):
                print(f"  {attr}")