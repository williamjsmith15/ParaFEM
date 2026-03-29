#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import parse_d_file, sanitize_zone_json

R_GAS = 8.314462618  # J/(mol*K)


def calculate_diffusivity(D0, Ea, T):
    """D(T) = D0 * exp(-Ea / (R * T))"""
    if T <= 0:
        return 0.0
    return D0 * math.exp(-Ea / (R_GAS * T))


def load_material_data(material_name):
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(os.path.dirname(base_dir), 'gdps_sievert_bc', 'materials.json')
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get(material_name)


def read_temperature_field(input_path):
    """Read node temperature values from an EnSight scalar file or .ini file.

    EnSight format: 4-line header (description, 'per node', 'part', 'coordinates').
    .ini format: first line is integer node count, then one value per line.
    """
    with open(input_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    header_lines = 4  # EnSight default
    try:
        int(lines[0])
        header_lines = 1  # .ini format: first line is count
    except (ValueError, IndexError):
        pass
    values = []
    for line in lines[header_lines:]:
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def parse_dat_params(dat_path):
    """Extract mesh parameters from a p123 or p124 .dat file.

    Works for both formats because the last 5 tokens on the integer-params line
    are always: nr, nip, nod, loaded_nodes, fixed_freedoms.
    p123 line has 7 tokens: nels nn nr nip nod loaded_nodes fixed_freedoms
    p124 line has 8 tokens: np_types nels nn nr nip nod loaded_nodes fixed_freedoms
    """
    element_type = 'hexahedron'
    nr = 0
    nip = 8
    loaded_nodes = 0
    fixed_freedoms = 0
    with open(dat_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in lines:
        if line.startswith("'"):
            element_type = line.strip("'").strip()
            continue
        parts = line.split()
        if len(parts) in (7, 8):
            try:
                vals = [int(p) for p in parts]
                nr = vals[-5]
                nip = vals[-4]
                loaded_nodes = vals[-2]
                fixed_freedoms = vals[-1]
            except ValueError:
                pass
    return element_type, nr, nip, loaded_nodes, fixed_freedoms


def main():
    parser = argparse.ArgumentParser(description="Compute temperature-dependent diffusivity field (.mat)")
    parser.add_argument('--mesh_d', required=True, help='Input .d mesh file')
    parser.add_argument('--dat_file', required=True, help='Input .dat control file (p123 or p124)')
    parser.add_argument('--temp_mode', choices=['scalar', 'field'], default='scalar')
    parser.add_argument('--temp_scalar', type=float, help='Temperature (K) if in scalar mode')
    parser.add_argument('--temp_field', help='Path to temperature field file (.ini or EnSight scalar)')
    parser.add_argument('--material', help='Material name from database')
    parser.add_argument('--D0', type=float, help='Diffusivity pre-exponential (m^2/s)')
    parser.add_argument('--Ea', type=float, help='Diffusivity activation energy (J/mol)')
    # p124 .dat parameters for the diffusion solve
    parser.add_argument('--nstep', type=int, default=1, help='Number of timesteps')
    parser.add_argument('--dtim', type=float, default=1e10, help='Timestep size (s); default is large for steady-state')
    parser.add_argument('--theta', type=float, default=1.0, help='Theta time integration parameter (1.0=fully implicit)')
    parser.add_argument('--val0', type=float, default=0.0, help='Initial concentration (mol/m^3)')
    parser.add_argument('--npri', type=int, default=1, help='Print every npri steps')
    parser.add_argument('--tol', type=float, default=1e-8, help='PCG convergence tolerance')
    parser.add_argument('--limit', type=int, default=200, help='PCG iteration limit')
    parser.add_argument('--nres', type=int, default=1, help='Monitor node index')
    parser.add_argument('--output_mat', required=True, help='Output .mat file')
    parser.add_argument('--output_d', required=True, help='Output modified .d file')
    parser.add_argument('--output_dat', required=True, help='Output p124 .dat file for diffusion solve')

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
    nodes, elements, element_type_d, nod = parse_d_file(args.mesh_d)
    nels = len(elements)
    nn = len(nodes)

    # Parse .dat parameters
    element_type, nr, nip, loaded_nodes, fixed_freedoms = parse_dat_params(args.dat_file)

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

    # Compute per-element diffusivity (average nodal temperature per element)
    element_diffusivity = {}
    for elem_id, elem_nodes in elements.items():
        temps = [node_temperatures.get(nid, 0.0) for nid in elem_nodes]
        avg_T = sum(temps) / len(temps) if temps else 0.0
        element_diffusivity[elem_id] = calculate_diffusivity(D0, Ea, avg_T)

    # Write .mat file (one material type per element, same D in all directions)
    print(f"Writing material data to {args.output_mat}")
    with open(args.output_mat, 'w') as f:
        f.write(f"*MATERIAL {nels} 5\n")
        f.write(" ID kx ky kz rho cp\n")
        for i in range(1, nels + 1):
            D = element_diffusivity.get(i, 0.0)
            f.write(f"{i:10d}  {D:16.8E}  {D:16.8E}  {D:16.8E}  {1.0:16.8E}  {1.0:16.8E}\n")

    # Write modified .d file (set etype_pp = elem_id for per-element material lookup)
    print(f"Writing modified mesh to {args.output_d}")
    with open(args.mesh_d, 'r') as fin, open(args.output_d, 'w') as fout:
        section = None
        for line in fin:
            stripped = line.strip()
            if stripped.startswith('*NODES'):
                section = 'nodes'
                fout.write(line)
                continue
            if stripped.startswith('*ELEMENTS'):
                section = 'elements'
                fout.write(line)
                continue
            if section == 'elements':
                parts = stripped.split()
                if not parts:
                    fout.write(line)
                    continue
                elem_id = int(parts[0])
                if len(parts) >= 4:
                    parts[3] = str(elem_id)
                    fout.write("  ".join(parts) + "\n")
                else:
                    fout.write(line)
            else:
                fout.write(line)

    # Write fresh p124 .dat for the diffusion solve
    # np_types = nels (one material type per element)
    print(f"Writing p124 control data to {args.output_dat}")
    with open(args.output_dat, 'w') as f:
        f.write(f"'{element_type}'\n")
        f.write("2\n")
        f.write("1\n")
        f.write(f"{nels:12d}{nels:12d}{nn:12d}{nr:12d}{nip:6d}{nod:6d}{loaded_nodes:12d}{fixed_freedoms:12d}\n")
        f.write(f"  {args.val0:.4E}  {args.dtim:.4E}  {args.nstep:d}  {args.npri:d}  {args.theta:.4f}  {args.tol:.4E}  {args.limit:d}  {args.nres:d}\n")

    print("Done.")


if __name__ == "__main__":
    main()
