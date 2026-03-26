#!/usr/bin/env python3
import json
import re
import numpy as np
from collections import defaultdict


def parse_d_file(filepath):
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
                nod_val = int(parts[2])
                node_ids = [int(x) for x in parts[4:4 + nod_val]]
                elements[elem_id] = node_ids

                if element_type is None:
                    nod = nod_val
                    element_type = 'hexahedron' if nod in (8, 20) else 'tetrahedron'

    return nodes, elements, element_type, nod


def find_boundary_nodes(nodes, elements, nod):
    if nod == 8:
        face_defs = [
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (2, 3, 7, 6),
            (0, 3, 7, 4), (1, 2, 6, 5),
        ]
    elif nod == 4:
        face_defs = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    elif nod == 20:
        face_defs = [
            (0, 1, 2, 3), (4, 5, 6, 7),
            (0, 1, 5, 4), (2, 3, 7, 6),
            (0, 3, 7, 4), (1, 2, 6, 5),
        ]
    else:
        return find_boundary_nodes_by_coords(nodes)

    face_count = defaultdict(int)
    face_to_nodes = {}

    for elem_id, elem_nodes in elements.items():
        for face_def in face_defs:
            face_nodes = tuple(sorted(elem_nodes[i] for i in face_def))
            face_count[face_nodes] += 1
            face_to_nodes[face_nodes] = set(elem_nodes[i] for i in face_def)

    boundary_nodes = set()
    for face_key, count in face_count.items():
        if count == 1:
            boundary_nodes.update(face_to_nodes[face_key])

    return boundary_nodes


def find_boundary_nodes_by_coords(nodes):
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
            zone_min = coord_min + zone['axis_min'] * coord_range
            zone_max = coord_min + zone['axis_max'] * coord_range
            if zone_min <= coord_val <= zone_max:
                fixed_nodes[node_id] = zone['temperature']
                break

    return fixed_nodes


def write_bnd(filepath, boundary_node_ids):
    with open(filepath, 'w') as f:
        for node_id in sorted(boundary_node_ids):
            f.write(f"{node_id:12d}     0\n")


def write_fix(filepath, fixed_nodes):
    with open(filepath, 'w') as f:
        for node_id in sorted(fixed_nodes.keys()):
            temp = fixed_nodes[node_id]
            f.write(f"{node_id:10d}         1  {temp:16.8E}\n")


def sanitize_zone_json(zone_str):
    zone_str = zone_str.replace("'", '"')
    zone_str = re.sub(r'(\b\w+\b)(\s*:)', r'"\1"\2', zone_str)
    return zone_str
