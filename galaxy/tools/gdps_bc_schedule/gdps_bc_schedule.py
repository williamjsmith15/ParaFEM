#!/usr/bin/env python3
"""
Generate a ParaFEM time-varying boundary condition schedule (.bcs).

Reads a ParaFEM .d file and a JSON schedule configuration to produce a .bcs
file that can be used by supporting solvers (e.g., gdps_thermal_transient)
to update BC values at specific time steps.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import (
    parse_d_file, find_boundary_nodes, assign_zones,
    write_fix, sanitize_zone_json, parse_nset_file,
    assign_nset_bc,
)


def write_bcs(filepath, schedule, nodes, boundary_nodes, bc_mode, bc_axis, nsets):
    """
    Writes a .bcs file from a schedule configuration.

    The key challenge is to ensure the node order in each .bcs block is
    identical to the order that the static gdps_bc_transient tool would
    produce. This is achieved by re-calculating the full set of fixed
    nodes for each step in the schedule.
    """
    with open(filepath, 'w') as f:
        for entry in schedule:
            step = entry['step']
            zones = entry['zones']

            if bc_mode == 'zone':
                fixed_nodes = assign_zones(nodes, boundary_nodes, zones, bc_axis)
            else:
                fixed_nodes = assign_nset_bc(nsets, zones)

            # Write step header
            f.write(str(step) + "\n")

            # Write data rows, sorted by node ID to match .fix file convention
            for node_id in sorted(fixed_nodes.keys()):
                temp = fixed_nodes[node_id]
                f.write(f"{node_id:10d}         1  {temp:16.8E}" + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate ParaFEM BC schedule file (.bcs)')
    parser.add_argument('--mesh_d', required=True)
    parser.add_argument('--schedule_config', required=True, help='Path to JSON schedule config file')
    parser.add_argument('--bc_mode', default='zone', choices=['zone', 'nset'], help='BC assignment mode')
    parser.add_argument('--nset_file', help='Input .nset file for nset mode')
    parser.add_argument('--bc_axis', default='z', choices=['x', 'y', 'z'])
    parser.add_argument('--output_bcs', required=True)
    args = parser.parse_args()

    # Parse schedule config (JSON string or file)
    if os.path.isfile(args.schedule_config):
        with open(args.schedule_config, 'r') as f:
            schedule = json.load(f)
    else:
        schedule_str = sanitize_zone_json(args.schedule_config)
        schedule = json.loads(schedule_str)

    print(f"Parsing mesh file: {args.mesh_d}")
    nodes, elements, element_type, nod = parse_d_file(args.mesh_d)
    print(f"  Nodes: {len(nodes)}, Elements: {len(elements)}")

    print(f"Finding boundary nodes...")
    boundary_nodes = find_boundary_nodes(nodes, elements, nod)
    print(f"  Total boundary nodes: {len(boundary_nodes)}")
    
    nsets = {}
    if args.bc_mode == 'nset':
        if not args.nset_file:
            print("Error: --nset_file is required for nset mode.")
            sys.exit(1)
        print(f"Parsing node sets from {args.nset_file}...")
        nsets = parse_nset_file(args.nset_file)

    print(f"Writing .bcs file to {args.output_bcs}...")
    write_bcs(args.output_bcs, schedule, nodes, boundary_nodes, args.bc_mode, args.bc_axis, nsets)

    print(f"Done. Wrote {len(schedule)} entries to schedule.")


if __name__ == '__main__':
    main()
