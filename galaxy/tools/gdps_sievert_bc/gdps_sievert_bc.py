#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import (
    parse_d_file, parse_nset_file, find_boundary_nodes, write_fix, sanitize_zone_json
)

R_GAS = 8.314462618  # J/(mol*K)


def calculate_solubility(S0, Es, T):
    """S(T) = S0 * exp(-Es / (R * T))"""
    if T <= 0:
        return 0.0
    return S0 * math.exp(-Es / (R_GAS * T))


def calculate_concentration(S0, Es, T, P):
    """C = S(T) * sqrt(P)"""
    if P < 0:
        return 0.0
    S = calculate_solubility(S0, Es, T)
    return S * math.sqrt(P)


def load_material_data(material_name):
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, 'materials.json')
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get(material_name)


def read_temperature_field(input_path):
    """Read node temperature values from an EnSight scalar file or .ini file.

    EnSight format: 4-line header (description, 'per node', 'part', 'coordinates').
    .ini format: first line is integer node count, then one value per line.
    """
    with open(input_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    header_lines = 4  # EnSight default
    try:
        int(lines[0])
        header_lines = 1  # .ini format
    except (ValueError, IndexError):
        pass
    values = []
    for line in lines[header_lines:]:
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def find_zone_nodes(nodes, boundary_nodes, axis, zone_min_frac, zone_max_frac):
    """Return boundary nodes whose normalised coordinate falls in [zone_min_frac, zone_max_frac]."""
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    all_coords = [nodes[nid][axis_idx] for nid in nodes]
    coord_min = min(all_coords)
    coord_max = max(all_coords)
    coord_range = coord_max - coord_min
    if coord_range == 0:
        return set()
    abs_min = coord_min + zone_min_frac * coord_range
    abs_max = coord_min + zone_max_frac * coord_range
    return {nid for nid in boundary_nodes if abs_min <= nodes[nid][axis_idx] <= abs_max}


def main():
    parser = argparse.ArgumentParser(description="Convert pressure BCs to concentration BCs using Sievert's Law")
    parser.add_argument('--mesh_d', required=True, help='Input .d mesh file')
    # NSET mode
    parser.add_argument('--nset_file', help='Input .nset file (required for nset bc_mode)')
    parser.add_argument('--upstream_nset', default='UPSTREAM_FACE', help='NSET name for upstream boundary')
    parser.add_argument('--downstream_nset', default='DOWNSTREAM_FACE', help='NSET name for downstream boundary')
    # Zone mode
    parser.add_argument('--bc_mode', choices=['nset', 'zone'], default='nset',
                        help='Node selection mode: nset (from .nset file) or zone (coordinate range)')
    parser.add_argument('--bc_axis', choices=['x', 'y', 'z'], default='z',
                        help='Axis for zone-based node selection')
    parser.add_argument('--upstream_min', type=float, default=0.0,
                        help='Upstream zone start (normalised 0-1 along bc_axis)')
    parser.add_argument('--upstream_max', type=float, default=0.05,
                        help='Upstream zone end (normalised 0-1 along bc_axis)')
    parser.add_argument('--downstream_min', type=float, default=0.95,
                        help='Downstream zone start (normalised 0-1 along bc_axis)')
    parser.add_argument('--downstream_max', type=float, default=1.0,
                        help='Downstream zone end (normalised 0-1 along bc_axis)')
    # Pressure inputs
    parser.add_argument('--upstream_p', type=float, help='Upstream pressure (Pa) for static mode')
    parser.add_argument('--downstream_p', type=float, help='Downstream pressure (Pa) for static mode')
    parser.add_argument('--pressure_schedule', help='JSON schedule for time-varying pressures')
    # Temperature
    parser.add_argument('--temp_mode', choices=['scalar', 'field'], default='scalar')
    parser.add_argument('--temp_scalar', type=float, help='Temperature (K) if in scalar mode')
    parser.add_argument('--temp_field', help='Path to temperature field file (.ini or EnSight scalar)')
    # Material
    parser.add_argument('--material', help='Material name from database')
    parser.add_argument('--S0', type=float, help='Solubility pre-exponential (mol/(m^3*Pa^0.5))')
    parser.add_argument('--Es', type=float, help='Solubility activation energy (J/mol)')
    # Outputs
    parser.add_argument('--output_fix', required=True, help='Output .fix file')
    parser.add_argument('--output_bcs', help='Output .bcs file (for scheduled pressure mode)')

    args = parser.parse_args()

    # Determine solubility parameters
    if args.material:
        mat_data = load_material_data(args.material)
        if not mat_data:
            print(f"Error: Material '{args.material}' not found in database.")
            sys.exit(1)
        S0 = mat_data['solubility']['S0']
        Es = mat_data['solubility']['Es']
    else:
        if args.S0 is None or args.Es is None:
            print("Error: Must provide either --material or both --S0 and --Es.")
            sys.exit(1)
        S0 = args.S0
        Es = args.Es

    # Parse mesh
    print(f"Parsing mesh: {args.mesh_d}")
    nodes, elements, element_type, nod = parse_d_file(args.mesh_d)

    # Identify upstream and downstream nodes
    if args.bc_mode == 'nset':
        if not args.nset_file:
            print("Error: --nset_file is required for nset bc_mode.")
            sys.exit(1)
        print(f"Parsing NSETs: {args.nset_file}")
        nsets = parse_nset_file(args.nset_file)
        upstream_nodes = nsets.get(args.upstream_nset, set())
        downstream_nodes = nsets.get(args.downstream_nset, set())
        if not upstream_nodes and not downstream_nodes:
            print(f"Warning: No nodes found in NSETs '{args.upstream_nset}' or '{args.downstream_nset}'.")
    else:
        boundary_nodes = find_boundary_nodes(nodes, elements, nod)
        upstream_nodes = find_zone_nodes(nodes, boundary_nodes, args.bc_axis,
                                         args.upstream_min, args.upstream_max)
        downstream_nodes = find_zone_nodes(nodes, boundary_nodes, args.bc_axis,
                                           args.downstream_min, args.downstream_max)
        print(f"Zone mode ({args.bc_axis}-axis): {len(upstream_nodes)} upstream, "
              f"{len(downstream_nodes)} downstream nodes.")

    # Handle temperature
    temperatures = {}
    if args.temp_mode == 'scalar':
        if args.temp_scalar is None:
            print("Error: --temp_scalar is required for scalar mode.")
            sys.exit(1)
        for node_id in upstream_nodes | downstream_nodes:
            temperatures[node_id] = args.temp_scalar
    else:
        if not args.temp_field:
            print("Error: --temp_field is required for field mode.")
            sys.exit(1)
        field_values = read_temperature_field(args.temp_field)
        for i, val in enumerate(field_values):
            node_id = i + 1
            if node_id in nodes:
                temperatures[node_id] = val

    def build_fixed_nodes(upstream_p, downstream_p):
        fixed = {}
        for node_id in upstream_nodes:
            T = temperatures.get(node_id)
            if T is not None:
                fixed[node_id] = calculate_concentration(S0, Es, T, upstream_p)
        for node_id in downstream_nodes:
            T = temperatures.get(node_id)
            if T is not None:
                fixed[node_id] = calculate_concentration(S0, Es, T, downstream_p)
        return fixed

    # Static case: write .fix using the provided pressures
    if args.upstream_p is not None and args.downstream_p is not None:
        fixed_nodes = build_fixed_nodes(args.upstream_p, args.downstream_p)
        print(f"Writing static BCs to {args.output_fix}")
        write_fix(args.output_fix, fixed_nodes)

    # Scheduled case: write .bcs and also .fix using the first schedule entry
    if args.pressure_schedule:
        if not args.output_bcs:
            print("Error: --output_bcs is required when using --pressure_schedule.")
            sys.exit(1)

        if os.path.isfile(args.pressure_schedule):
            with open(args.pressure_schedule, 'r') as f:
                schedule = json.load(f)
        else:
            schedule = json.loads(sanitize_zone_json(args.pressure_schedule))

        # Write .fix from first schedule entry for initial conditions
        first = schedule[0]
        fixed_nodes = build_fixed_nodes(first['upstream_p'], first['downstream_p'])
        print(f"Writing initial BCs (.fix) from schedule step {first['step']} to {args.output_fix}")
        write_fix(args.output_fix, fixed_nodes)

        # Write .bcs schedule
        print(f"Writing BC schedule to {args.output_bcs}")
        with open(args.output_bcs, 'w') as f:
            for entry in schedule:
                step = entry['step']
                up_p = entry['upstream_p']
                down_p = entry['downstream_p']
                f.write(f"{step}\n")
                current_fixed = build_fixed_nodes(up_p, down_p)
                for node_id in sorted(current_fixed.keys()):
                    val = current_fixed[node_id]
                    f.write(f"{node_id:10d}         1  {val:16.8E}\n")

    print("Done.")


if __name__ == "__main__":
    main()
