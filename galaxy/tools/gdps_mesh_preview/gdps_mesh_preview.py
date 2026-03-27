#!/usr/bin/env python3
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Add parent dir to path for parafem_common
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import parse_d_file, parse_nset_file

# VTK cell type IDs
VTK_CELL_TYPES = {
    4: 10,   # VTK_TETRA
    8: 12,   # VTK_HEXAHEDRON
    10: 24,  # VTK_QUADRATIC_TETRA
    20: 25,  # VTK_QUADRATIC_HEXAHEDRON
}

# ParaFEM (Abaqus-convention, mesh=2) to VTK node ordering
NODE_REORDER = {
    4: None,
    8: None,
    10: None,
    20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15],
}

def build_vtu_tree(nodes, elements, nod, point_data=None, cell_data=None):
    """Build an ElementTree for a VTU file."""
    sorted_node_ids = sorted(nodes.keys())
    sorted_elem_ids = sorted(elements.keys())

    nn = len(sorted_node_ids)
    nels = len(sorted_elem_ids)
    node_id_map = {nid: idx for idx, nid in enumerate(sorted_node_ids)}

    vtk_cell_type = VTK_CELL_TYPES.get(nod, 12)
    reorder = NODE_REORDER.get(nod)

    root = ET.Element('VTKFile', type='UnstructuredGrid', version='1.0', byte_order='LittleEndian')
    grid = ET.SubElement(root, 'UnstructuredGrid')
    piece = ET.SubElement(grid, 'Piece', NumberOfPoints=str(nn), NumberOfCells=str(nels))

    points_el = ET.SubElement(piece, 'Points')
    coords_arr = ET.SubElement(points_el, 'DataArray', type='Float64', NumberOfComponents='3', format='ascii')
    coord_lines = []
    for nid in sorted_node_ids:
        c = nodes[nid]
        coord_lines.append(f"{c[0]:.10E} {c[1]:.10E} {c[2]:.10E}")
    coords_arr.text = '\n' + '\n'.join(coord_lines) + '\n'

    cells_el = ET.SubElement(piece, 'Cells')
    conn_arr = ET.SubElement(cells_el, 'DataArray', type='Int32', Name='connectivity', format='ascii')
    conn_lines = []
    for eid in sorted_elem_ids:
        # elements[eid] in parafem_common is just a list of node IDs
        elem_nodes = elements[eid]
        if reorder:
            elem_nodes = [elem_nodes[i] for i in reorder]
        indices = [str(node_id_map[nid]) for nid in elem_nodes]
        conn_lines.append(' '.join(indices))
    conn_arr.text = '\n' + '\n'.join(conn_lines) + '\n'

    offsets_arr = ET.SubElement(cells_el, 'DataArray', type='Int32', Name='offsets', format='ascii')
    offsets = [str(nod * (i + 1)) for i in range(nels)]
    offsets_arr.text = '\n' + ' '.join(offsets) + '\n'

    types_arr = ET.SubElement(cells_el, 'DataArray', type='UInt8', Name='types', format='ascii')
    types_arr.text = '\n' + ' '.join([str(vtk_cell_type)] * nels) + '\n'

    if point_data:
        pd_el = ET.SubElement(piece, 'PointData')
        for name, data in point_data.items():
            arr = ET.SubElement(pd_el, 'DataArray', type='Float64', Name=name, format='ascii')
            arr.text = '\n' + ' '.join(f"{v:.10E}" for v in data) + '\n'

    return root

def write_vtu(root, filepath):
    rough = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(rough)
    with open(filepath, 'w') as f:
        f.write(dom.toprettyxml(indent='  '))

def main():
    parser = argparse.ArgumentParser(description='Generate NSET preview VTU')
    parser.add_argument('--mesh_d', required=True, help='ParaFEM .d file')
    parser.add_argument('--nset_file', required=True, help='ParaFEM .nset file')
    parser.add_argument('--output', required=True, help='Output .vtu file')
    args = parser.parse_args()

    print(f"Reading mesh: {args.mesh_d}")
    nodes, elements, _, nod = parse_d_file(args.mesh_d)
    print(f"Reading NSETs: {args.nset_file}")
    nsets = parse_nset_file(args.nset_file)

    sorted_node_ids = sorted(nodes.keys())
    node_id_map = {nid: idx for idx, nid in enumerate(sorted_node_ids)}

    # Initialize SurfaceID with 0 (no NSET)
    surface_ids = [0.0] * len(sorted_node_ids)

    # Map NSET names to numeric IDs (1-based)
    nset_list = sorted(nsets.keys())
    print(f"Found {len(nset_list)} node sets:")
    for idx, name in enumerate(nset_list):
        s_id = idx + 1
        node_count = len(nsets[name])
        print(f"  [{s_id}] {name} ({node_count} nodes)")
        for node_id in nsets[name]:
            node_idx = node_id_map.get(node_id)
            if node_idx is not None:
                surface_ids[node_idx] = float(s_id)

    root = build_vtu_tree(nodes, elements, nod, point_data={'SurfaceID': surface_ids})
    write_vtu(root, args.output)
    print(f"Wrote preview VTU to {args.output}")

if __name__ == '__main__':
    main()
