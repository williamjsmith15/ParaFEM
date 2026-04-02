#!/usr/bin/env python3
"""
Compute downstream permeation flux from a FEM diffusion solution.

Reads the concentration field, mesh geometry, and per-element diffusivity,
then integrates J = -D * dC/dn over the downstream face to give the total
gas flow rate in units comparable to mass spectrometer output.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
import tarfile
import tempfile
import shutil


R_GAS = 8.314462618  # J/(mol·K)


def parse_d_file(filepath):
    nodes = {}
    elements = {}
    nod = 0
    section = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*THREE_DIMENSIONAL'):
                continue
            if line == '*NODES':
                section = 'nodes'
                continue
            if line == '*ELEMENTS':
                section = 'elements'
                continue
            if line.startswith('*'):
                continue
            if section == 'nodes':
                parts = line.split()
                nodes[int(parts[0])] = [float(x) for x in parts[1:4]]
            elif section == 'elements':
                parts = line.split()
                elem_id = int(parts[0])
                nod_val = int(parts[2])
                node_ids = [int(x) for x in parts[4:4 + nod_val]]
                elements[elem_id] = node_ids
                if nod == 0:
                    nod = nod_val
    return nodes, elements, nod


def parse_mat_file(filepath):
    """Parse .mat file — returns {elem_id: D} using kx column."""
    D_per_elem = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('ID'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    D_per_elem[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue
    return D_per_elem


def parse_ensi_scalar(filepath):
    values = []
    with open(filepath, 'r') as f:
        for _ in range(4):
            next(f)
        for line in f:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    continue
    return values


def find_ensi_files(directory, jobname):
    found = {}
    for path in sorted(glob.glob(os.path.join(directory, f"{jobname}.ensi.*"))):
        m = re.match(rf'{re.escape(jobname)}\.ensi\.([A-Z]+)-(\d+)',
                     os.path.basename(path))
        if m:
            found.setdefault(m.group(1), []).append((int(m.group(2)), path))
    return found


def parse_ensi_case(directory, jobname):
    """Parse .ensi.case and return {step_number: simulation_time}."""
    case_path = os.path.join(directory, f"{jobname}.ensi.case")
    if not os.path.exists(case_path):
        return {}

    step_times = {}
    in_time = False
    start_num = 1
    increment = 1
    n_steps = 0
    time_values = []

    with open(case_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'TIME':
                in_time = True
                continue
            if not in_time:
                continue
            if line.startswith('number of steps'):
                n_steps = int(line.split(':')[1].strip())
            elif line.startswith('filename start number'):
                start_num = int(line.split(':')[1].strip())
            elif line.startswith('filename increment'):
                increment = int(line.split(':')[1].strip())
            elif line.startswith('time values'):
                continue
            elif in_time and line and not line.startswith('time set'):
                try:
                    time_values.extend(float(v) for v in line.split())
                except ValueError:
                    pass

    for i, t in enumerate(time_values):
        step_num = start_num + i * increment
        step_times[step_num] = t

    return step_times


def quad_area(coords):
    """Area of a planar quad via two triangle cross products."""
    p = coords
    a = [p[1][i] - p[0][i] for i in range(3)]
    b = [p[2][i] - p[0][i] for i in range(3)]
    c = [p[3][i] - p[0][i] for i in range(3)]
    cross1 = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
    cross2 = [b[1]*c[2] - b[2]*c[1], b[2]*c[0] - b[0]*c[2], b[0]*c[1] - b[1]*c[0]]
    area1 = 0.5 * math.sqrt(sum(x**2 for x in cross1))
    area2 = 0.5 * math.sqrt(sum(x**2 for x in cross2))
    return area1 + area2


def compute_downstream_flux(nodes, elements, D_per_elem, conc, axis, face, tol):
    ax = {'x': 0, 'y': 1, 'z': 2}[axis]
    all_ax = [c[ax] for c in nodes.values()]
    face_coord = max(all_ax) if face == 'max' else min(all_ax)

    sorted_node_ids = sorted(nodes.keys())
    node_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}

    total_flux = 0.0
    n_face_elements = 0

    for elem_id, node_ids in elements.items():
        D = D_per_elem.get(elem_id, 0.0)
        if D == 0.0:
            continue

        face_nodes = [nid for nid in node_ids
                      if abs(nodes[nid][ax] - face_coord) <= tol]
        interior_nodes = [nid for nid in node_ids if nid not in face_nodes]

        if len(face_nodes) != 4 or len(interior_nodes) != 4:
            continue

        C_face = sum(conc[node_idx[nid]] for nid in face_nodes) / 4.0
        C_interior = sum(conc[node_idx[nid]] for nid in interior_nodes) / 4.0

        face_ax_mean = sum(nodes[nid][ax] for nid in face_nodes) / 4.0
        int_ax_mean = sum(nodes[nid][ax] for nid in interior_nodes) / 4.0
        dz = abs(face_ax_mean - int_ax_mean)
        if dz == 0.0:
            continue

        area = quad_area([nodes[nid] for nid in face_nodes])

        # J = -D * dC/dn, outward normal: sign depends on face
        grad_C = (C_face - C_interior) / dz
        total_flux += -D * grad_C * area
        n_face_elements += 1

    return total_flux, n_face_elements


def main():
    parser = argparse.ArgumentParser(
        description='Compute downstream permeation flux from FEM diffusion solution')
    parser.add_argument('--mesh_d', required=True)
    parser.add_argument('--ensi_tarball', required=True)
    parser.add_argument('--mat_file', required=True)
    parser.add_argument('--jobname', default='job')
    parser.add_argument('--axis', default='z', choices=['x', 'y', 'z'])
    parser.add_argument('--face', default='max', choices=['min', 'max'])
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--temp_ambient', type=float, default=298.15,
                        help='Gas measurement temperature (K) for mol/s to Pa.m3/s conversion')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print(f"Parsing mesh: {args.mesh_d}")
    nodes, elements, nod = parse_d_file(args.mesh_d)
    nn = len(nodes)
    print(f"  Nodes: {nn}, Elements: {len(elements)}, Nodes/element: {nod}")

    print(f"Parsing diffusivity: {args.mat_file}")
    D_per_elem = parse_mat_file(args.mat_file)
    D_vals = list(D_per_elem.values())
    print(f"  D range: {min(D_vals):.4E} — {max(D_vals):.4E} m2/s")

    # Face area (bounding box of downstream face nodes) — same for all steps
    ax = {'x': 0, 'y': 1, 'z': 2}[args.axis]
    all_ax = [c[ax] for c in nodes.values()]
    face_coord = max(all_ax) if args.face == 'max' else min(all_ax)
    face_node_coords = [c for c in nodes.values() if abs(c[ax] - face_coord) <= args.tol]
    other = [i for i in range(3) if i != ax]
    span = [(max(c[a] for c in face_node_coords) - min(c[a] for c in face_node_coords))
            for a in other]
    face_area = span[0] * span[1]

    tmpdir = tempfile.mkdtemp()
    try:
        with tarfile.open(args.ensi_tarball, 'r:*') as tar:
            tar.extractall(tmpdir)

        step_times = parse_ensi_case(tmpdir, args.jobname)
        ensi_files = find_ensi_files(tmpdir, args.jobname)

        field_label = None
        all_steps = None
        for var_type in ('NDTTR', 'NDPTL'):
            if var_type in ensi_files:
                field_label = var_type
                all_steps = sorted(ensi_files[var_type])
                break

        if all_steps is None:
            print("Error: no NDTTR or NDPTL field found in tarball.", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(all_steps)} timestep(s) of {field_label}")
        print(f"Computing flux on {args.face} face along {args.axis}-axis...")

        rows = []
        for step_num, path in all_steps:
            conc_field = parse_ensi_scalar(path)
            if len(conc_field) != nn:
                print(f"  Warning: step {step_num} has {len(conc_field)} values, "
                      f"expected {nn} — skipping", file=sys.stderr)
                continue

            total_mol_s, n_elems = compute_downstream_flux(
                nodes, elements, D_per_elem, conc_field, args.axis, args.face, args.tol)

            flux_pa_m3_s = total_mol_s * R_GAS * args.temp_ambient
            flux_mbar_l_s = flux_pa_m3_s * 10.0
            flux_density = total_mol_s / face_area if face_area > 0 else 0.0
            sim_time = step_times.get(step_num, float('nan'))

            rows.append({
                'step': step_num,
                'time_s': sim_time,
                'flux_mol_s': total_mol_s,
                'flux_pa_m3_s': flux_pa_m3_s,
                'flux_mbar_l_s': flux_mbar_l_s,
                'flux_density_mol_m2_s': flux_density,
            })
            print(f"  Step {step_num:6d}  t={sim_time:.4E} s  "
                  f"J={flux_mbar_l_s:.4E} mbar.L/s")

    finally:
        shutil.rmtree(tmpdir)

    fieldnames = ['step', 'time_s', 'flux_mol_s', 'flux_pa_m3_s',
                  'flux_mbar_l_s', 'flux_density_mol_m2_s']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} row(s) to {args.output}")
    print(f"Face: {args.face} along {args.axis}-axis  |  Area: {face_area:.6E} m2  |  "
          f"Field: {field_label}  |  Ambient: {args.temp_ambient:.2f} K")


if __name__ == '__main__':
    main()
