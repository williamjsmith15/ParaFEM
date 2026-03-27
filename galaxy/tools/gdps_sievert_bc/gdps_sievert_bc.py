#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

# Import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import (
    parse_d_file, parse_nset_file, write_fix, sanitize_zone_json
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
    """Read temperature values from an EnSight scalar file."""
    values = []
    header_skipped = 0
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if header_skipped < 4:
                header_skipped += 1
                continue
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
    return values

def main():
    parser = argparse.ArgumentParser(description="Convert pressure BCs to concentration BCs using Sievert's Law")
    parser.add_argument('--mesh_d', required=True, help='Input .d mesh file')
    parser.add_argument('--nset_file', required=True, help='Input .nset file')
    parser.add_argument('--upstream_p', type=float, help='Upstream pressure (Pa)')
    parser.add_argument('--downstream_p', type=float, help='Downstream pressure (Pa)')
    parser.add_argument('--pressure_schedule', help='JSON schedule for time-varying pressures')
    parser.add_argument('--temp_mode', choices=['scalar', 'field'], default='scalar', help='Temperature mode')
    parser.add_argument('--temp_scalar', type=float, help='Temperature (K) if in scalar mode')
    parser.add_argument('--temp_field', help='Path to EnSight temperature field file if in field mode')
    parser.add_argument('--material', help='Material name from database')
    parser.add_argument('--S0', type=float, help='Solubility pre-exponential (mol/(m^3*Pa^0.5))')
    parser.add_argument('--Es', type=float, help='Solubility activation energy (J/mol)')
    parser.add_argument('--upstream_nset', default='UPSTREAM_FACE', help='NSET name for upstream boundary')
    parser.add_argument('--downstream_nset', default='DOWNSTREAM_FACE', help='NSET name for downstream boundary')
    parser.add_argument('--output_fix', help='Output .fix file (for static BC)')
    parser.add_argument('--output_bcs', help='Output .bcs file (for scheduled BC)')

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

    # Parse mesh and NSETs
    print(f"Parsing mesh: {args.mesh_d}")
    nodes, _, _, _ = parse_d_file(args.mesh_d)
    print(f"Parsing NSETs: {args.nset_file}")
    nsets = parse_nset_file(args.nset_file)

    upstream_nodes = nsets.get(args.upstream_nset, set())
    downstream_nodes = nsets.get(args.downstream_nset, set())

    if not upstream_nodes and not downstream_nodes:
        print(f"Warning: No nodes found in NSETs '{args.upstream_nset}' or '{args.downstream_nset}'.")

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

    # Static case
    if args.upstream_p is not None and args.downstream_p is not None:
        fixed_nodes = {}
        for node_id in upstream_nodes:
            T = temperatures.get(node_id)
            if T is not None:
                fixed_nodes[node_id] = calculate_concentration(S0, Es, T, args.upstream_p)
        for node_id in downstream_nodes:
            T = temperatures.get(node_id)
            if T is not None:
                fixed_nodes[node_id] = calculate_concentration(S0, Es, T, args.downstream_p)
        
        if args.output_fix:
            print(f"Writing static BCs to {args.output_fix}")
            write_fix(args.output_fix, fixed_nodes)

    # Scheduled case
    if args.pressure_schedule:
        if not args.output_bcs:
            print("Error: --output_bcs is required when using --pressure_schedule.")
            sys.exit(1)
        
        if os.path.isfile(args.pressure_schedule):
            with open(args.pressure_schedule, 'r') as f:
                schedule = json.load(f)
        else:
            schedule = json.loads(sanitize_zone_json(args.pressure_schedule))
        
        print(f"Writing BC schedule to {args.output_bcs}")
        with open(args.output_bcs, 'w') as f:
            for entry in schedule:
                step = entry['step']
                up_p = entry['upstream_p']
                down_p = entry['downstream_p']
                
                f.write(f"{step}\n")
                
                # We need to sort by node ID to match ParaFEM convention
                current_fixed = {}
                for node_id in upstream_nodes:
                    T = temperatures.get(node_id)
                    if T is not None:
                        current_fixed[node_id] = calculate_concentration(S0, Es, T, up_p)
                for node_id in downstream_nodes:
                    T = temperatures.get(node_id)
                    if T is not None:
                        current_fixed[node_id] = calculate_concentration(S0, Es, T, down_p)
                
                for node_id in sorted(current_fixed.keys()):
                    val = current_fixed[node_id]
                    f.write(f"{node_id:10d}         1  {val:16.8E}\n")
    
    print("Done.")

if __name__ == "__main__":
    main()
