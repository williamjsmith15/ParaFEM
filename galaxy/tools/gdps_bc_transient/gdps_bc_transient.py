#!/usr/bin/env python3
"""
Generate ParaFEM boundary condition and material files for transient thermal analysis (p124).

Reads a ParaFEM .d file, identifies boundary surface nodes, and assigns
Dirichlet (fixed temperature) BCs to nodes within user-defined heater zones.
Also writes the .mat material properties file required by p124.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import (
    parse_d_file, find_boundary_nodes, assign_zones,
    write_bnd, write_fix, sanitize_zone_json,
)


def write_dat(filepath, nels, nn, nr, nod, fixed_freedoms,
              val0, dtim, nstep, npri, theta, tol, limit, element_type, np_types=1):
    nip = 8 if nod == 8 else 27 if nod == 20 else 1
    nres = 1
    loaded_freedoms = 0

    with open(filepath, 'w') as f:
        f.write(f"'{element_type}'\n")
        f.write(f"2\n")   # meshgen = 2 (Abaqus numbering)
        f.write(f"1\n")   # partition = 1 (internal)
        f.write(f"{np_types:12d}{nels:12d}{nn:12d}{nr:12d}{nip:6d}{nod:6d}"
                f"{loaded_freedoms:12d}{fixed_freedoms:12d}\n")
        f.write(f"  {val0:.4E}  {dtim:.4E}{nstep:8d}{npri:8d}"
                f"  {theta:.4f}  {tol:.4E}{limit:8d}{nres:8d}\n")


def write_mat(filepath, kx, ky, kz, rho, cp, np_types=1):
    with open(filepath, 'w') as f:
        f.write(f"*MATERIAL {np_types} 5\n")
        f.write(f" ID kx ky kz rho cp\n")
        for i in range(1, np_types + 1):
            f.write(f"{i}  {kx:.6E}  {ky:.6E}  {kz:.6E}  {rho:.6E}  {cp:.6E}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate ParaFEM thermal BCs and material file for p124')
    parser.add_argument('--mesh_d', required=True)
    parser.add_argument('--kx', type=float, required=True)
    parser.add_argument('--ky', type=float, required=True)
    parser.add_argument('--kz', type=float, required=True)
    parser.add_argument('--rho', type=float, required=True)
    parser.add_argument('--cp', type=float, required=True)
    parser.add_argument('--val0', type=float, required=True)
    parser.add_argument('--dtim', type=float, required=True)
    parser.add_argument('--nstep', type=int, required=True)
    parser.add_argument('--npri', type=int, required=True)
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--zone_config', required=True, help='Path to JSON zone config file')
    parser.add_argument('--bc_axis', default='z', choices=['x', 'y', 'z'])
    parser.add_argument('--output_dat', required=True)
    parser.add_argument('--output_bnd', required=True)
    parser.add_argument('--output_fix', required=True)
    parser.add_argument('--output_mat', required=True)
    parser.add_argument('--output_d', required=True)
    args = parser.parse_args()

    with open(args.zone_config) as f:
        zone_str = f.read()
    zone_str = sanitize_zone_json(zone_str)
    zones = json.loads(zone_str)

    print(f"Parsing mesh file: {args.mesh_d}")
    nodes, elements, element_type, nod = parse_d_file(args.mesh_d)
    nn = len(nodes)
    nels = len(elements)
    print(f"  Nodes: {nn}, Elements: {nels}, Type: {element_type}, Nodes/element: {nod}")

    print(f"Finding boundary nodes...")
    boundary_nodes = find_boundary_nodes(nodes, elements, nod)
    print(f"  Total boundary nodes: {len(boundary_nodes)}")

    print(f"Assigning heater zones along {args.bc_axis}-axis...")
    fixed_nodes = assign_zones(nodes, boundary_nodes, zones, args.bc_axis)
    print(f"  Nodes with fixed temperature: {len(fixed_nodes)}")
    for zone in zones:
        zone_nodes = sum(1 for nid, t in fixed_nodes.items() if t == zone['temperature'])
        print(f"    Zone [{zone['axis_min']:.2f}, {zone['axis_max']:.2f}] (normalized): "
              f"{zone_nodes} nodes at {zone['temperature']} K")

    nr = 0
    fixed_freedoms = len(fixed_nodes)

    print(f"Writing output files...")
    # Write boundary nodes to .bnd for visualization (solver ignores it when nr=0)
    write_bnd(args.output_bnd, boundary_nodes)
    write_fix(args.output_fix, fixed_nodes)
    write_dat(args.output_dat, nels, nn, nr, nod, fixed_freedoms,
              args.val0, args.dtim, args.nstep, args.npri, args.theta,
              args.tol, args.limit, element_type)
    write_mat(args.output_mat, args.kx, args.ky, args.kz, args.rho, args.cp)
    shutil.copy2(args.mesh_d, args.output_d)

    print(f"Done. nr={nr} restrained, {fixed_freedoms} fixed freedoms via penalty method.")


if __name__ == '__main__':
    main()
