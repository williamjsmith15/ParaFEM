#!/usr/bin/env python3
"""
Generate ParaFEM boundary condition files for steady-state thermal analysis (p123).

Reads a ParaFEM .d file, identifies boundary surface nodes, and assigns
Dirichlet (fixed temperature) BCs to nodes within user-defined heater zones.
"""

import argparse
import json
import re
import shutil
import numpy as np
from collections import defaultdict


def parse_d_file(filepath):
    """Parse a ParaFEM .d file to extract nodes and elements."""
    nodes = {}
    elements = {}
    element_type = None
    nod = 0

    section = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('*THREE_DIMENSIONAL'):
                continue
            if line == '*NODES':
                section = 'nodes'
                continue
            if line == '*ELEMENTS':
                section = 'elements'
                continue

            if section == 'nodes':
                parts = line.split()
                node_id = int(parts[0])
                coords = [float(x) for x in parts[1:4]]
                nodes[node_id] = np.array(coords)

            elif section == 'elements':
                parts = line.split()
                elem_id = int(parts[0])
                # Format: elem_id ndim nod nip node1 node2 ... nodeN mat_id
                ndim_val = int(parts[1])
                nod_val = int(parts[2])
                nip_val = int(parts[3])
                node_ids = [int(x) for x in parts[4:4 + nod_val]]
                mat_id = int(parts[4 + nod_val])
                elements[elem_id] = node_ids

                if element_type is None:
                    nod = nod_val
                    if nod in (8,):
                        element_type = 'hexahedron'
                    elif nod in (20,):
                        element_type = 'hexahedron'
                    elif nod in (4,):
                        element_type = 'tetrahedron'
                    elif nod in (10,):
                        element_type = 'tetrahedron'
                    else:
                        element_type = 'hexahedron'

    return nodes, elements, element_type, nod


def find_boundary_nodes(nodes, elements, nod):
    """
    Find boundary nodes by identifying element faces that appear only once.
    A face shared by two elements is internal; a face on only one element is a boundary face.
    """
    if nod == 8:
        # 8-node hex face definitions (local node indices, 0-based)
        face_defs = [
            (0, 1, 2, 3),  # bottom
            (4, 5, 6, 7),  # top
            (0, 1, 5, 4),  # front
            (2, 3, 7, 6),  # back
            (0, 3, 7, 4),  # left
            (1, 2, 6, 5),  # right
        ]
    elif nod == 4:
        # 4-node tet face definitions
        face_defs = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ]
    elif nod == 20:
        # 20-node hex: use corner nodes only for face identification
        face_defs = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (0, 3, 7, 4),
            (1, 2, 6, 5),
        ]
    else:
        # Fallback: use coordinate-based boundary detection
        return find_boundary_nodes_by_coords(nodes)

    face_count = defaultdict(int)
    face_to_nodes = {}

    for elem_id, elem_nodes in elements.items():
        for face_def in face_defs:
            if nod == 20:
                face_nodes = tuple(sorted(elem_nodes[i] for i in face_def))
            else:
                face_nodes = tuple(sorted(elem_nodes[i] for i in face_def))
            face_count[face_nodes] += 1
            face_to_nodes[face_nodes] = set(elem_nodes[i] for i in face_def)

    boundary_nodes = set()
    for face_key, count in face_count.items():
        if count == 1:
            boundary_nodes.update(face_to_nodes[face_key])

    return boundary_nodes


def find_boundary_nodes_by_coords(nodes):
    """
    Fallback: identify boundary nodes by coordinate extremes.
    Nodes on any face of the bounding box are boundary nodes.
    """
    coords = np.array(list(nodes.values()))
    node_ids = list(nodes.keys())

    tol = 1e-10
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    boundary = set()
    for i, nid in enumerate(node_ids):
        c = coords[i]
        for dim in range(3):
            if abs(c[dim] - mins[dim]) < tol or abs(c[dim] - maxs[dim]) < tol:
                boundary.add(nid)
                break

    return boundary


def assign_zones(nodes, boundary_nodes, zones, axis):
    """
    Assign boundary nodes to heater zones based on their coordinate along the specified axis.
    Zone axis_min/axis_max are normalized fractions (0.0 = mesh min, 1.0 = mesh max).
    Returns dict of {node_id: temperature} for nodes in zones.
    """
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]

    all_coords = np.array([nodes[nid][axis_idx] for nid in nodes])
    coord_min = float(all_coords.min())
    coord_max = float(all_coords.max())
    coord_range = coord_max - coord_min

    print(f"  Mesh {axis}-axis range: [{coord_min:.6f}, {coord_max:.6f}]")

    fixed_nodes = {}
    for node_id in boundary_nodes:
        coord_val = nodes[node_id][axis_idx]
        for zone in zones:
            # Map normalized [0,1] fractions to actual mesh coordinates
            zone_min = coord_min + zone['axis_min'] * coord_range
            zone_max = coord_min + zone['axis_max'] * coord_range
            if zone_min <= coord_val <= zone_max:
                fixed_nodes[node_id] = zone['temperature']
                break

    return fixed_nodes


def write_bnd(filepath, fixed_node_ids):
    """Write .bnd file: restrained nodes with nodof=1 format."""
    with open(filepath, 'w') as f:
        for node_id in sorted(fixed_node_ids):
            f.write(f"{node_id:12d}     0\n")


def write_fix(filepath, fixed_nodes):
    """Write .fix file: node_id sense value (3-column format for read_fixed)."""
    with open(filepath, 'w') as f:
        for node_id in sorted(fixed_nodes.keys()):
            temp = fixed_nodes[node_id]
            f.write(f"{node_id:10d}         1  {temp:16.8E}\n")


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
    parser.add_argument('--bc_axis', default='z', choices=['x', 'y', 'z'])
    parser.add_argument('--output_dat', required=True)
    parser.add_argument('--output_bnd', required=True)
    parser.add_argument('--output_fix', required=True)
    parser.add_argument('--output_d', required=True)
    args = parser.parse_args()

    # Parse zone config (JSON string or file)
    # Galaxy's shell quoting can strip quotes from JSON keys, leaving
    # {axis_min: 0.0} instead of {"axis_min": 0.0}. Fix with regex.
    zone_str = args.zone_config.replace("'", '"')
    zone_str = re.sub(r'(\b\w+\b)(\s*:)', r'"\1"\2', zone_str)
    try:
        zones = json.loads(zone_str)
    except json.JSONDecodeError:
        with open(args.zone_config, 'r') as f:
            zones = json.load(f)

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
    # Write empty .bnd (nr=0, no restrained nodes — all BCs via penalty method in .fix)
    with open(args.output_bnd, 'w') as f:
        pass
    write_fix(args.output_fix, fixed_nodes)
    write_dat(args.output_dat, nels, nn, nr, nod, fixed_freedoms,
              args.kx, args.ky, args.kz, args.tol, args.limit, element_type)

    # Pass through the .d file
    shutil.copy2(args.mesh_d, args.output_d)

    print(f"Done. nr={nr} restrained, {fixed_freedoms} fixed freedoms via penalty method.")


if __name__ == '__main__':
    main()
