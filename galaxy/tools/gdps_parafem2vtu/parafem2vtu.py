#!/usr/bin/env python3
"""
Convert ParaFEM output files to VTK Unstructured Grid (.vtu) format.

Reads the mesh (.d file), boundary conditions (.bnd, .fix), and solver output
(EnSight scalar/vector fields) and writes a single .vtu file for ParaView.

For time-varying results (e.g. p124 NDTTR), writes a .pvd collection file
referencing per-timestep .vtu files.

Supported solver output types:
  NDPTL  — scalar per node (potentials, p123)
  NDTTR  — scalar per node, time-varying (temperatures, p124)
  DISPL  — vector per node (displacements, p121/p122/p129/p1210)
  EIGV   — vector per node (eigenvectors, p128)
"""

import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom


# VTK cell type IDs
VTK_CELL_TYPES = {
    4: 10,   # VTK_TETRA
    8: 12,   # VTK_HEXAHEDRON
    10: 24,  # VTK_QUADRATIC_TETRA
    20: 25,  # VTK_QUADRATIC_HEXAHEDRON
}

# ParaFEM (Abaqus-convention, mesh=2) to VTK node ordering
# ParaFEM 8-node hex uses Abaqus ordering which matches VTK
NODE_REORDER = {
    4: None,
    8: None,
    10: None,
    20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15],
}


def parse_d_file(filepath):
    """Parse a ParaFEM .d file. Returns nodes dict, elements dict, nod."""
    nodes = {}
    elements = {}
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
                nodes[node_id] = coords

            elif section == 'elements':
                parts = line.split()
                elem_id = int(parts[0])
                nod_val = int(parts[2])
                node_ids = [int(x) for x in parts[4:4 + nod_val]]
                mat_id = int(parts[4 + nod_val])
                elements[elem_id] = {'nodes': node_ids, 'mat_id': mat_id}
                if nod == 0:
                    nod = nod_val

    return nodes, elements, nod


def parse_ensi_scalar(filepath):
    """Parse an EnSight Gold scalar-per-node file. Returns list of floats."""
    values = []
    with open(filepath, 'r') as f:
        # Skip 4 header lines
        for _ in range(4):
            next(f)
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return values


def parse_ensi_vector(filepath, nn):
    """Parse an EnSight Gold vector-per-node file. Returns list of [x,y,z]."""
    values = []
    with open(filepath, 'r') as f:
        for _ in range(4):
            next(f)
        raw = []
        for line in f:
            line = line.strip()
            if line:
                raw.append(float(line))

    # EnSight stores vectors as: all x, then all y, then all z
    if len(raw) == 3 * nn:
        for i in range(nn):
            values.append([raw[i], raw[nn + i], raw[2 * nn + i]])
    else:
        # Fallback: interleaved
        for i in range(0, len(raw), 3):
            values.append(raw[i:i + 3])

    return values


def parse_bnd_file(filepath, nn):
    """Parse .bnd file. Returns per-node flag array (1=restrained, 0=free)."""
    flags = [0.0] * nn
    if not os.path.exists(filepath):
        return flags
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                node_id = int(parts[0])
                if 1 <= node_id <= nn:
                    flags[node_id - 1] = 1.0
    return flags


def parse_fix_file(filepath, nn):
    """Parse .fix file. Returns per-node value array (NaN where not fixed)."""
    values = [float('nan')] * nn
    if not os.path.exists(filepath):
        return values
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                val = float(parts[2])
                if 1 <= node_id <= nn:
                    values[node_id - 1] = val
    return values


def find_ensi_files(directory, jobname):
    """Find all EnSight variable files for a job. Returns dict of {type: [paths]}."""
    found = {}
    pattern = os.path.join(directory, f"{jobname}.ensi.*")
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        # Match patterns like job.ensi.NDPTL-000001
        m = re.match(rf'{re.escape(jobname)}\.ensi\.([A-Z]+)-(\d+)', basename)
        if m:
            var_type = m.group(1)
            found.setdefault(var_type, []).append(path)
    return found


def build_vtu_tree(nodes, elements, nod, point_data=None, cell_data=None):
    """Build an ElementTree for a VTU file."""
    # Sort nodes and elements by ID for consistent ordering
    sorted_node_ids = sorted(nodes.keys())
    sorted_elem_ids = sorted(elements.keys())

    nn = len(sorted_node_ids)
    nels = len(sorted_elem_ids)

    # Build node ID -> 0-based index mapping
    node_id_map = {nid: idx for idx, nid in enumerate(sorted_node_ids)}

    vtk_cell_type = VTK_CELL_TYPES.get(nod, 12)
    reorder = NODE_REORDER.get(nod)

    # Root
    root = ET.Element('VTKFile', type='UnstructuredGrid', version='1.0',
                       byte_order='LittleEndian')
    grid = ET.SubElement(root, 'UnstructuredGrid')
    piece = ET.SubElement(grid, 'Piece',
                          NumberOfPoints=str(nn), NumberOfCells=str(nels))

    # Points
    points_el = ET.SubElement(piece, 'Points')
    coords_arr = ET.SubElement(points_el, 'DataArray', type='Float64',
                                NumberOfComponents='3', format='ascii')
    coord_lines = []
    for nid in sorted_node_ids:
        c = nodes[nid]
        coord_lines.append(f"{c[0]:.10E} {c[1]:.10E} {c[2]:.10E}")
    coords_arr.text = '\n' + '\n'.join(coord_lines) + '\n'

    # Cells
    cells_el = ET.SubElement(piece, 'Cells')

    # Connectivity
    conn_arr = ET.SubElement(cells_el, 'DataArray', type='Int32',
                              Name='connectivity', format='ascii')
    conn_lines = []
    for eid in sorted_elem_ids:
        elem_nodes = elements[eid]['nodes']
        if reorder:
            elem_nodes = [elem_nodes[i] for i in reorder]
        # Convert to 0-based indices
        indices = [str(node_id_map[nid]) for nid in elem_nodes]
        conn_lines.append(' '.join(indices))
    conn_arr.text = '\n' + '\n'.join(conn_lines) + '\n'

    # Offsets
    offsets_arr = ET.SubElement(cells_el, 'DataArray', type='Int32',
                                 Name='offsets', format='ascii')
    offsets = [str(nod * (i + 1)) for i in range(nels)]
    offsets_arr.text = '\n' + ' '.join(offsets) + '\n'

    # Types
    types_arr = ET.SubElement(cells_el, 'DataArray', type='UInt8',
                               Name='types', format='ascii')
    types_arr.text = '\n' + ' '.join([str(vtk_cell_type)] * nels) + '\n'

    # Point data
    if point_data:
        pd_el = ET.SubElement(piece, 'PointData')
        for name, data in point_data.items():
            if isinstance(data[0], list):
                # Vector data
                arr = ET.SubElement(pd_el, 'DataArray', type='Float64',
                                    Name=name, NumberOfComponents='3',
                                    format='ascii')
                lines = [f"{v[0]:.10E} {v[1]:.10E} {v[2]:.10E}" for v in data]
                arr.text = '\n' + '\n'.join(lines) + '\n'
            else:
                # Scalar data
                arr = ET.SubElement(pd_el, 'DataArray', type='Float64',
                                    Name=name, format='ascii')
                arr.text = '\n' + ' '.join(f"{v:.10E}" for v in data) + '\n'

    # Cell data
    if cell_data:
        cd_el = ET.SubElement(piece, 'CellData')
        for name, data in cell_data.items():
            arr = ET.SubElement(cd_el, 'DataArray', type='Int32',
                                Name=name, format='ascii')
            arr.text = '\n' + ' '.join(str(int(v)) for v in data) + '\n'

    return root


def write_vtu(root, filepath):
    """Write a VTU ElementTree to file with pretty formatting."""
    rough = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(rough)
    with open(filepath, 'w') as f:
        f.write(dom.toprettyxml(indent='  '))


def write_pvd(timestep_files, filepath):
    """Write a PVD collection file referencing per-timestep VTU files."""
    root = ET.Element('VTKFile', type='Collection', version='1.0',
                       byte_order='LittleEndian')
    collection = ET.SubElement(root, 'Collection')
    for i, vtu_path in enumerate(timestep_files):
        ET.SubElement(collection, 'DataSet',
                      timestep=str(i), part='0',
                      file=os.path.basename(vtu_path))
    write_vtu(root, filepath)


# Mapping of EnSight variable types to human-readable names
ENSI_VAR_NAMES = {
    'NDPTL': 'Potential',
    'NDTTR': 'Temperature',
    'DISPL': 'Displacement',
    'EIGV': 'Eigenvector',
}

ENSI_VAR_IS_VECTOR = {
    'NDPTL': False,
    'NDTTR': False,
    'DISPL': True,
    'EIGV': True,
}


def main():
    parser = argparse.ArgumentParser(
        description='Convert ParaFEM output to VTK Unstructured Grid (.vtu)')
    parser.add_argument('--mesh_d', required=True, help='ParaFEM .d mesh file')
    parser.add_argument('--bnd', default=None, help='ParaFEM .bnd file (optional)')
    parser.add_argument('--fix', default=None, help='ParaFEM .fix file (optional)')
    parser.add_argument('--ensi_dir', default=None,
                        help='Directory containing .ensi.* solver output files')
    parser.add_argument('--ensi_tarball', default=None,
                        help='Tarball of .ensi.* files (alternative to --ensi_dir)')
    parser.add_argument('--jobname', default='job',
                        help='Job name prefix for EnSight files')
    parser.add_argument('--output', required=True,
                        help='Output .vtu file (or .pvd for time-varying)')
    args = parser.parse_args()

    print(f"Parsing mesh: {args.mesh_d}")
    nodes, elements, nod = parse_d_file(args.mesh_d)
    nn = len(nodes)
    nels = len(elements)
    sorted_elem_ids = sorted(elements.keys())
    print(f"  Nodes: {nn}, Elements: {nels}, Nodes/element: {nod}")

    # Collect point data
    point_data = {}
    cell_data = {}

    # Boundary node flags
    if args.bnd:
        bnd_flags = parse_bnd_file(args.bnd, nn)
        if any(v > 0 for v in bnd_flags):
            point_data['BoundaryNodes'] = bnd_flags
            print(f"  Boundary nodes: {sum(1 for v in bnd_flags if v > 0)}")

    # Fixed freedom values
    if args.fix:
        fix_vals = parse_fix_file(args.fix, nn)
        if any(v == v for v in fix_vals):  # at least one non-NaN
            point_data['FixedFreedoms'] = fix_vals
            print(f"  Fixed freedoms: {sum(1 for v in fix_vals if v == v)}")

    # Material IDs (from element data)
    mat_ids = [elements[eid]['mat_id'] for eid in sorted_elem_ids]
    if any(m != mat_ids[0] for m in mat_ids):
        cell_data['MaterialID'] = mat_ids

    # Unpack tarball if provided
    ensi_dir = args.ensi_dir
    if args.ensi_tarball and os.path.exists(args.ensi_tarball):
        import tarfile
        ensi_dir = '/tmp/ensi_unpack'
        os.makedirs(ensi_dir, exist_ok=True)
        with tarfile.open(args.ensi_tarball, 'r:gz') as tf:
            tf.extractall(ensi_dir)
        print(f"  Unpacked EnSight tarball to {ensi_dir}")

    # Find and load solver output fields
    if ensi_dir:
        ensi_files = find_ensi_files(ensi_dir, args.jobname)
        print(f"  Found EnSight variables: {list(ensi_files.keys())}")

        for var_type, paths in ensi_files.items():
            var_name = ENSI_VAR_NAMES.get(var_type, var_type)
            is_vector = ENSI_VAR_IS_VECTOR.get(var_type, False)

            if len(paths) == 1:
                # Single timestep — add directly to point data
                if is_vector:
                    data = parse_ensi_vector(paths[0], nn)
                else:
                    data = parse_ensi_scalar(paths[0])

                if len(data) == nn:
                    point_data[var_name] = data
                    print(f"  Loaded {var_name}: {len(data)} values")
                else:
                    print(f"  WARNING: {var_name} has {len(data)} values, "
                          f"expected {nn}. Skipping.", file=sys.stderr)

            elif len(paths) > 1:
                # Multiple timesteps — write .pvd + per-step .vtu files
                output_base = os.path.splitext(args.output)[0]
                pvd_path = output_base + '.pvd'
                vtu_paths = []

                for step_idx, path in enumerate(paths):
                    if is_vector:
                        data = parse_ensi_vector(path, nn)
                    else:
                        data = parse_ensi_scalar(path)

                    step_point_data = dict(point_data)
                    if len(data) == nn:
                        step_point_data[var_name] = data

                    step_vtu = f"{output_base}_{step_idx:06d}.vtu"
                    root = build_vtu_tree(nodes, elements, nod,
                                          step_point_data, cell_data)
                    write_vtu(root, step_vtu)
                    vtu_paths.append(step_vtu)
                    print(f"  Wrote timestep {step_idx}: {step_vtu}")

                write_pvd(vtu_paths, pvd_path)
                print(f"  Wrote PVD collection: {pvd_path}")
                print(f"Done. {len(vtu_paths)} timestep files + PVD collection.")
                return

    # Single timestep — write one .vtu
    root = build_vtu_tree(nodes, elements, nod, point_data, cell_data)
    write_vtu(root, args.output)
    print(f"Done. Wrote {args.output}")


if __name__ == '__main__':
    main()
