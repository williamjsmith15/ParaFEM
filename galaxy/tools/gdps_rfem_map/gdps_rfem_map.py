#!/usr/bin/env python3
import argparse
import os
import sys
import tarfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from parafem_common import parse_d_file


def read_rf_mat(mat_path):
    """Parse *MATERIAL header, return {iel: D} dict.
    Supports plain .mat files and .tar.gz archives containing a .mat file.
    """
    if tarfile.is_tarfile(mat_path):
        with tarfile.open(mat_path, "r:gz") as tar:
            mat_member = None
            for m in tar.getmembers():
                if m.name.endswith('.mat'):
                    mat_member = m
                    break
            if not mat_member:
                raise ValueError(f"No .mat file found in archive: {mat_path}")
            with tar.extractfile(mat_member) as f:
                content = f.read().decode().splitlines()
    else:
        with open(mat_path, 'r') as f:
            content = f.readlines()

    rf_mat = {}
    lines = [l.strip() for l in content if l.strip()]

    if not lines or not lines[0].startswith('*MATERIAL'):
        raise ValueError(f"Invalid .mat file header in: {mat_path}")

    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 2:
            iel = int(parts[0])
            val = float(parts[1])
            rf_mat[iel] = val
    return rf_mat


def parse_dat_file(dat_path):
    """Parse rfemsolve .dat file (free-form Fortran multi-line).
    Returns dict of all scalar parameters.
    """
    with open(dat_path, 'r') as f:
        raw = f.read()

    tokens = raw.split()
    if len(tokens) < 14:
        raise ValueError(f"dat file has too few tokens ({len(tokens)}): {dat_path}")

    return {
        'element':          tokens[0].strip("'\""),
        'mesh':             int(tokens[1]),
        'partition':        int(tokens[2]),
        'np_types':         int(tokens[3]),
        'nels':             int(tokens[4]),
        'nn':               int(tokens[5]),
        'nr':               int(tokens[6]),
        'nip':              int(tokens[7]),
        'nod':              int(tokens[8]),
        'loaded_freedoms':  int(tokens[9]),
        'fixed_freedoms':   int(tokens[10]),
        'tol':              float(tokens[11]),
        'limit':            int(tokens[12]),
        'mises':            float(tokens[13]),
    }


def write_dat_file(path, p, nels, nn):
    """Write rfemsolve .dat with updated nels/nn/np_types, preserving all other fields."""
    with open(path, 'w') as f:
        f.write(f"'{p['element']}'\n")
        f.write(f"{p['mesh']}\n")
        f.write(f"{p['partition']}\n")
        f.write(f"{nels}\n")  # np_types = nels for per-element material
        f.write(f"{nels:8d} {nn:8d} {p['nr']:8d} {p['nip']:8d} {p['nod']:8d} "
                f"{p['loaded_freedoms']:8d} {p['fixed_freedoms']:8d}\n")
        f.write(f"{p['tol']:14.6E} {p['limit']:8d} {p['mises']:14.6E}\n")


def get_centroid_mapping(nxe, nye, nze, aa, bb, cc, origin):
    """Return function: centroid (x,y,z) -> RF element index (1-based).
    Element ordering: z outer, y middle, x inner (matches rfemfield_thermal output).
    """
    def centroid_to_rf_iel(cx, cy, cz):
        lx = cx - origin[0]
        ly = cy - origin[1]
        lz = cz - origin[2]
        ix = min(max(0, int(lx / aa)), nxe - 1)
        iy = min(max(0, int(ly / bb)), nye - 1)
        iz = min(max(0, int(lz / cc)), nze - 1)
        return (iz * nye + iy) * nxe + ix + 1
    return centroid_to_rf_iel


def main():
    parser = argparse.ArgumentParser(description="Map RFEM field onto arbitrary target mesh")
    parser.add_argument('--rf_mat',      required=True,  help='Reference .mat file from rfemfield_thermal')
    parser.add_argument('--target_d',    required=True,  help='Target mesh .d file')
    parser.add_argument('--dat_file',    required=True,  help='Control .dat from rfembc_thermal (provides fixed_freedoms)')
    parser.add_argument('--instance_id', default='001',  help='Instance ID suffix')
    parser.add_argument('--nxe',  type=int,   required=True, help='RF grid elements in X')
    parser.add_argument('--nze',  type=int,   required=True, help='RF grid elements in Z')
    parser.add_argument('--aa',   type=float, required=True, help='RF element size X (m)')
    parser.add_argument('--bb',   type=float, required=True, help='RF element size Y (m)')
    parser.add_argument('--cc',   type=float, required=True, help='RF element size Z (m)')
    parser.add_argument('--origin_x', type=float, default=0.0)
    parser.add_argument('--origin_y', type=float, default=0.0)
    parser.add_argument('--origin_z', type=float, default=0.0)
    parser.add_argument('--output_tar', required=True, help='Output tar.gz archive')
    parser.add_argument('--output_res', required=True, help='Output mapping statistics')

    args = parser.parse_args()

    rf_mat = read_rf_mat(args.rf_mat)
    nels_rf = len(rf_mat)
    nye = nels_rf // (args.nxe * args.nze)
    print(f"RF Grid: {args.nxe} x {nye} x {args.nze} (total {nels_rf})")

    nodes, elements, element_type, nod = parse_d_file(args.target_d)
    m_nels = len(elements)
    m_nn   = len(nodes)
    print(f"Target Mesh: {m_nels} elements, {m_nn} nodes, type={element_type}")

    dat_params = parse_dat_file(args.dat_file)

    origin = [args.origin_x, args.origin_y, args.origin_z]
    map_func = get_centroid_mapping(args.nxe, nye, args.nze, args.aa, args.bb, args.cc, origin)

    mapped_vals = {}
    out_of_bounds = 0

    for iel, node_ids in elements.items():
        coords = np.array([nodes[nid] for nid in node_ids])
        centroid = coords.mean(axis=0)

        lx = centroid[0] - origin[0]
        ly = centroid[1] - origin[1]
        lz = centroid[2] - origin[2]
        if (lx < 0 or lx > args.nxe * args.aa or
                ly < 0 or ly > nye * args.bb or
                lz < 0 or lz > args.nze * args.cc):
            out_of_bounds += 1

        rf_iel = map_func(*centroid)
        mapped_vals[iel] = rf_mat.get(rf_iel, rf_mat[1])

    target_base = os.path.splitext(os.path.basename(args.target_d))[0]
    base_name   = f"{target_base}-{args.instance_id}"
    d_out   = f"{base_name}.d"
    dat_out = f"{base_name}.dat"
    mat_out = f"{base_name}.mat"

    # .d file — copy target mesh with per-element material IDs
    with open(args.target_d, 'r') as fin, open(d_out, 'w') as fout:
        section = None
        for line in fin:
            stripped = line.strip()
            if not stripped:
                fout.write("\n")
                continue
            if stripped == '*NODES':
                section = 'nodes'
            if stripped == '*ELEMENTS':
                section = 'elements'

            if section == 'elements' and not stripped.startswith('*'):
                parts = stripped.split()
                if len(parts) >= 4:
                    iel = int(parts[0])
                    parts[3] = str(iel)
                    parts[-1] = str(iel)
                    fout.write("  ".join(parts) + "\n")
                    continue
            fout.write(line)

    # .mat file
    with open(mat_out, 'w') as f:
        f.write(f"*MATERIAL {m_nels} 1\n")
        f.write("kx\n")
        for iel in range(1, m_nels + 1):
            f.write(f"{iel:8d} {mapped_vals[iel]:16.8E}\n")

    # .dat file — updated nels/nn/np_types, fixed_freedoms preserved from input
    write_dat_file(dat_out, dat_params, m_nels, m_nn)

    with tarfile.open(args.output_tar, "w:gz") as tar:
        tar.add(d_out)
        tar.add(dat_out)
        tar.add(mat_out)

    with open(args.output_res, 'w') as f:
        f.write(f"RFEM Mapping Results for instance {args.instance_id}\n")
        f.write(f"Target elements: {m_nels}\n")
        f.write(f"Out of bounds centroids: {out_of_bounds}\n")
        d_vals = list(mapped_vals.values())
        f.write(f"Mapped D min: {min(d_vals):.4e}\n")
        f.write(f"Mapped D max: {max(d_vals):.4e}\n")
        f.write(f"Mapped D mean: {np.mean(d_vals):.4e}\n")
        f.write(f"Fixed freedoms (from BC dat): {dat_params['fixed_freedoms']}\n")

    os.remove(d_out)
    os.remove(dat_out)
    os.remove(mat_out)


if __name__ == "__main__":
    main()
