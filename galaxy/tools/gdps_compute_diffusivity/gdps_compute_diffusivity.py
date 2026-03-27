#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

# Import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import (
    parse_d_file, sanitize_zone_json
)

R_GAS = 8.314462618  # J/(mol*K)

def calculate_diffusivity(D0, Ea, T):
    """D(T) = D0 * exp(-Ea / (R * T))"""
    if T <= 0:
        return 0.0
    return D0 * math.exp(-Ea / (R_GAS * T))

def load_material_data(material_name):
    base_dir = os.path.dirname(__file__)
    # materials.json is shared with gdps_sievert_bc
    json_path = os.path.join(os.path.dirname(base_dir), 'gdps_sievert_bc', 'materials.json')
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
    parser = argparse.ArgumentParser(description="Compute temperature-dependent diffusivity field (.mat)")
    parser.add_argument('--mesh_d', required=True, help='Input .d mesh file')
    parser.add_argument('--dat_file', required=True, help='Input .dat control file')
    parser.add_argument('--temp_mode', choices=['scalar', 'field'], default='scalar')
    parser.add_argument('--temp_scalar', type=float, help='Temperature (K) if in scalar mode')
    parser.add_argument('--temp_field', help='Path to EnSight temperature field file if in field mode')
    parser.add_argument('--material', help='Material name from database')
    parser.add_argument('--D0', type=float, help='Diffusivity pre-exponential (m^2/s)')
    parser.add_argument('--Ea', type=float, help='Diffusivity activation energy (J/mol)')
    parser.add_argument('--output_mat', required=True, help='Output .mat file')
    parser.add_argument('--output_d', required=True, help='Output modified .d file')
    parser.add_argument('--output_dat', required=True, help='Output modified .dat file')

    args = parser.parse_args()

    # Determine diffusivity parameters
    if args.material:
        mat_data = load_material_data(args.material)
        if not mat_data:
            print(f"Error: Material '{args.material}' not found in database.")
            sys.exit(1)
        D0 = mat_data['diffusivity']['D0']
        Ea = mat_data['diffusivity']['Ea']
    else:
        if args.D0 is None or args.Ea is None:
            print("Error: Must provide either --material or both --D0 and --Ea.")
            sys.exit(1)
        D0 = args.D0
        Ea = args.Ea

    # Parse mesh
    print(f"Parsing mesh: {args.mesh_d}")
    nodes, elements, element_type, nod = parse_d_file(args.mesh_d)
    nels = len(elements)
    nn = len(nodes)

    # Handle temperature
    node_temperatures = {}
    if args.temp_mode == 'scalar':
        if args.temp_scalar is None:
            print("Error: --temp_scalar is required for scalar mode.")
            sys.exit(1)
        for node_id in nodes:
            node_temperatures[node_id] = args.temp_scalar
    else:
        if not args.temp_field:
            print("Error: --temp_field is required for field mode.")
            sys.exit(1)
        field_values = read_temperature_field(args.temp_field)
        for i, val in enumerate(field_values):
            node_id = i + 1
            if node_id in nodes:
                node_temperatures[node_id] = val

    # Compute per-element diffusivity
    element_diffusivity = {}
    for elem_id, elem_nodes in elements.items():
        # Compute average temperature of nodes in the element
        temps = [node_temperatures.get(nid, 0.0) for nid in elem_nodes]
        if temps:
            avg_T = sum(temps) / len(temps)
        else:
            avg_T = 0.0
        element_diffusivity[elem_id] = calculate_diffusivity(D0, Ea, avg_T)

    # Write .mat file
    print(f"Writing material data to {args.output_mat}")
    with open(args.output_mat, 'w') as f:
        # p124 style mat file
        f.write(f"*MATERIAL {nels} 5\n")
        f.write("id kx ky kz rho cp\n")
        for i in range(1, nels + 1):
            D = element_diffusivity.get(i, 0.0)
            f.write(f"{i:10d}  {D:16.8E}  {D:16.8E}  {D:16.8E}  {1.0:16.8E}  {1.0:16.8E}\n")

    # Write modified .d file
    print(f"Writing modified mesh to {args.output_d}")
    with open(args.mesh_d, 'r') as fin, open(args.output_d, 'w') as fout:
        section = None
        for line in fin:
            if line.startswith('*NODES'):
                section = 'nodes'
                fout.write(line)
                continue
            if line.startswith('*ELEMENTS'):
                section = 'elements'
                fout.write(line)
                continue
            
            if section == 'elements':
                parts = line.split()
                if not parts:
                    fout.write(line)
                    continue
                elem_id = int(parts[0])
                # ParaFEM .d element line: id, type_code, nodes_per_elem, etype_pp, nodes...
                # Actually, some versions skip etype_pp?
                # Let's re-verify the .d format. 
                # p124 expects: iel, i_type, nod_val, etype_pp, g_num_pp(:,iel)
                # i_type is usually 1, nod_val is 8 or 20.
                if len(parts) >= 4:
                    # Update 4th column (etype_pp) to be elem_id
                    parts[3] = str(elem_id)
                    fout.write("  ".join(parts) + "\n")
                else:
                    fout.write(line)
            else:
                fout.write(line)

    # Write modified .dat file
    print(f"Writing modified control data to {args.output_dat}")
    with open(args.dat_file, 'r') as fin, open(args.output_dat, 'w') as fout:
        # Read the p124 .dat line
        for line in fin:
            if line.startswith("'"): # element_type line
                fout.write(line)
                continue
            parts = line.split()
            if len(parts) >= 19: # p124 .dat has ~19 values
                # element, mesh, partition, np_types, nels, nn, nr, nip, nod, ...
                # Wait, the first value is element_type which might be on its own line
                # Let's assume the first line is the element type and the second is the numbers
                if len(parts) > 10:
                    # np_types is the 4th value? No, 1st is mesh, 2nd is partition, 3rd is np_types?
                    # Let's check read_p124 again.
                    # READ(10,*) element,mesh,partition,np_types,nels,nn,nr,nip,nod, ...
                    # Actually READ(10,*) element is on the SAME line if it's not a character.
                    # But it IS a character: element.
                    # Usually: 'hexahedron' 1 1 1 8 27 1 8 0 ...
                    # Let's try to find np_types. It's the 4th value after 'element'.
                    parts[3] = str(nels)
                    fout.write("  ".join(parts) + "\n")
                else:
                    fout.write(line)
            else:
                fout.write(line)

    print("Done.")

if __name__ == "__main__":
    main()
