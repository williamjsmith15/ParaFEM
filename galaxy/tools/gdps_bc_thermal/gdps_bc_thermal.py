#!/usr/bin/env python3
"""
Generate ParaFEM boundary condition files for steady-state thermal analysis (p123).

Reads a ParaFEM .d file, identifies boundary surface nodes, and assigns
Dirichlet (fixed temperature) BCs to nodes within user-defined heater zones.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import (
    parse_d_file, find_boundary_nodes, assign_zones,
    write_bnd, write_fix, sanitize_zone_json, parse_nset_file,
    assign_nset_bc,
)


def write_dat(filepath, nels, nn, nr, nod, fixed_freedoms, kx, ky, kz, tol, limit, element_type):
    """Write .dat file in p123 format."""
    nip = 8 if nod == 8 else 27 if nod == 20 else 1
    nres = 1

    with open(filepath, 'w') as f:
        f.write(f"'{element_type}'\n")
        f.write("2\n")  # Abaqus node numbering
        f.write("1\n")  # Internal partitioning
        f.write(f"{nels:12d}{nn:12d}{nr:12d}{nip:6d}{nod:6d}{0:12d}{fixed_freedoms:12d}\n")
        f.write(f"  {kx:.4E}  {ky:.4E}  {kz:.4E}  {tol:.4E}{limit:8d}{nres:8d}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate ParaFEM thermal BCs for p123')
    parser.add_argument('--mesh_d', required=True, help='Input .d mesh file')
    parser.add_argument('--kx', type=float, required=True)
    parser.add_argument('--ky', type=float, required=True)
    parser.add_argument('--kz', type=float, required=True)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--zone_config', required=True, help='JSON zone config string or file path')
    parser.add_argument('--bc_mode', default='zone', choices=['zone', 'nset'], help='BC assignment mode')
    parser.add_argument('--nset_file', help='Input .nset file for nset mode')
    parser.add_argument('--bc_axis', default='z', choices=['x', 'y', 'z'])
    parser.add_argument('--output_dat', required=True)
    parser.add_argument('--output_bnd', required=True)
    parser.add_argument('--output_fix', required=True)
    parser.add_argument('--output_d', required=True)
    args = parser.parse_args()

    # Parse zone config (JSON string or file)
    if os.path.isfile(args.zone_config):
        with open(args.zone_config, 'r') as f:
            zone_str = sanitize_zone_json(f.read())
        zones = json.loads(zone_str)
    else:
        zone_str = sanitize_zone_json(args.zone_config)
        zones = json.loads(zone_str)

    print(f"Parsing mesh file: {args.mesh_d}")
    nodes, elements, element_type, nod = parse_d_file(args.mesh_d)
    nn = len(nodes)
    nels = len(elements)
    print(f"  Nodes: {nn}, Elements: {nels}, Type: {element_type}, Nodes/element: {nod}")

    print(f"Finding boundary nodes...")
    boundary_nodes = find_boundary_nodes(nodes, elements, nod)
    print(f"  Total boundary nodes: {len(boundary_nodes)}")

    fixed_nodes = {}
    if args.bc_mode == 'zone':
        print(f"Assigning heater zones along {args.bc_axis}-axis...")
        fixed_nodes = assign_zones(nodes, boundary_nodes, zones, args.bc_axis)
        print(f"  Nodes with fixed temperature: {len(fixed_nodes)}")
        for zone in zones:
            zone_nodes = sum(1 for nid, t in fixed_nodes.items() if t == zone['temperature'])
            print(f"    Zone [{zone['axis_min']:.2f}, {zone['axis_max']:.2f}] (normalized): "
                  f"{zone_nodes} nodes at {zone['temperature']} K")
    else:
        if not args.nset_file:
            print("Error: --nset_file is required for nset mode.")
            sys.exit(1)
        print(f"Assigning BCs using node sets from {args.nset_file}...")
        nsets = parse_nset_file(args.nset_file)
        fixed_nodes = assign_nset_bc(nsets, zones)

    nr = 0
    fixed_freedoms = len(fixed_nodes)

    print(f"Writing output files...")
    # Write boundary nodes to .bnd for visualization (solver ignores it when nr=0)
    write_bnd(args.output_bnd, boundary_nodes)
    write_fix(args.output_fix, fixed_nodes)
    write_dat(args.output_dat, nels, nn, nr, nod, fixed_freedoms,
              args.kx, args.ky, args.kz, args.tol, args.limit, element_type)

    # Pass through the .d file
    shutil.copy2(args.mesh_d, args.output_d)

    print(f"Done. nr={nr} restrained, {fixed_freedoms} fixed freedoms via penalty method.")


if __name__ == '__main__':
    main()
