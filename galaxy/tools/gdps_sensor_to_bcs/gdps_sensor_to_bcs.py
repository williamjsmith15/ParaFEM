#!/usr/bin/env python3
"""
Convert InfluxDB sensor CSV to a ParaFEM time-varying BC schedule (.bcs).

Reads a sensor CSV with an elapsed_seconds column plus temperature columns,
and maps each row to a time step index using the provided dtim value.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import (
    parse_d_file, find_boundary_nodes, assign_zones,
    sanitize_zone_json, parse_nset_file, assign_nset_bc,
)


def write_bcs_from_sensor(filepath, rows, zone_template, nodes, boundary_nodes,
                          bc_mode, bc_axis, nsets, dtim):
    # Group sensor readings by simulation step, accumulating values for averaging
    col_names = [entry['col_name'] for entry in zone_template]
    step_accum = {}
    for row in rows:
        elapsed = float(row['elapsed_seconds'])
        step = round(elapsed / dtim)
        if step == 0:
            continue
        if step not in step_accum:
            step_accum[step] = {col: [] for col in col_names}
        for col in col_names:
            step_accum[step][col].append(float(row[col]))

    with open(filepath, 'w') as f:
        for step in sorted(step_accum.keys()):
            col_avgs = {col: sum(vals) / len(vals) for col, vals in step_accum[step].items()}

            zones = []
            for entry in zone_template:
                col = entry['col_name']
                zone = {k: v for k, v in entry.items() if k != 'col_name'}
                zone['temperature'] = col_avgs[col]
                zones.append(zone)

            if bc_mode == 'zone':
                fixed_nodes = assign_zones(nodes, boundary_nodes, zones, bc_axis)
            else:
                fixed_nodes = assign_nset_bc(nsets, zones)

            f.write(str(step) + "\n")
            for node_id in sorted(fixed_nodes.keys()):
                temp = fixed_nodes[node_id]
                f.write(f"{node_id:10d}         1  {temp:16.8E}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert sensor CSV to ParaFEM .bcs file')
    parser.add_argument('--mesh_d', required=True)
    parser.add_argument('--sensor_csv', required=True)
    parser.add_argument('--zone_template', required=True, help='Path to JSON zone template file')
    parser.add_argument('--bc_mode', default='zone', choices=['zone', 'nset'])
    parser.add_argument('--bc_axis', default='z', choices=['x', 'y', 'z'])
    parser.add_argument('--nset_file')
    parser.add_argument('--dtim', type=float, default=1.0)
    parser.add_argument('--output_bcs', required=True)
    args = parser.parse_args()

    with open(args.zone_template, 'r') as f:
        content = f.read()
    template_str = sanitize_zone_json(content)
    zone_template = json.loads(template_str)

    print(f"Parsing mesh file: {args.mesh_d}")
    nodes, elements, element_type, nod = parse_d_file(args.mesh_d)
    print(f"  Nodes: {len(nodes)}, Elements: {len(elements)}")

    print("Finding boundary nodes...")
    boundary_nodes = find_boundary_nodes(nodes, elements, nod)
    print(f"  Total boundary nodes: {len(boundary_nodes)}")

    nsets = {}
    if args.bc_mode == 'nset':
        if not args.nset_file:
            print("Error: --nset_file is required for nset mode.")
            sys.exit(1)
        print(f"Parsing node sets from {args.nset_file}...")
        nsets = parse_nset_file(args.nset_file)

    with open(args.sensor_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} rows from sensor CSV.")

    print(f"Writing .bcs file to {args.output_bcs}...")
    write_bcs_from_sensor(
        args.output_bcs, rows, zone_template, nodes, boundary_nodes,
        args.bc_mode, args.bc_axis, nsets, args.dtim,
    )
    print("Done.")


if __name__ == '__main__':
    main()
