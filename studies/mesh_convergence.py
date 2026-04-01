"""RFEM Mesh Convergence Study.

Verifies that the deterministic (uniform D) downstream flux converges
to the analytical solution J = D * C_up / L as the mesh is refined.

Usage:
    python studies/mesh_convergence.py
Outputs (studies/results/):
    mesh_convergence.csv
    mesh_convergence.png
"""

import os
import subprocess
import tarfile
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'galaxy', 'tools'))
from parafem_common import parse_d_file

IMAGE_RFEM = 'williamjsmith15/parafem-rfem:local'

# GDPS membrane parameters (from GDPS_CASE_STUDY.md)
D    = 1.5e-12   # diffusivity [m^2/s]
C_UP = 4.5e-3    # upstream concentration [mol/m^3]
L    = 5e-3      # membrane lateral size [m]
T    = 0.5e-3    # membrane thickness [m]


def docker_run(workdir, image, command):
    r = subprocess.run(
        ['docker', 'run', '--rm', '-v', f'{workdir}:/work', '-w', '/work', image, 'bash', '-c', command],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"STDOUT: {r.stdout}")
        print(f"STDERR: {r.stderr}")
    return r


def write_rfem_rf(workdir, nels, n, aa, bb, cc, emn, esd):
    # Correlation lengths: ~2x element size to avoid extreme ratios
    thx = 2 * aa
    thy = 2 * bb
    thz = 2 * cc
    content = (
        f"sim3de\n1\n{nels} {n} {n}\n"
        f"{aa} {bb} {cc}\n{thx} {thy} {thz}\n"
        f"{emn} {esd}\ndlavx3\n0.3\n"
    )
    with open(os.path.join(workdir, 'rfem.rf'), 'w') as f:
        f.write(content)


def extract_flux(workdir, jobname, inst_id, dz):
    """Extract downstream flux J = D * avg(dC/dz) from EnSight output."""
    ndptl_path = os.path.join(workdir, f"{jobname}-{inst_id}.ensi.NDPTL-000001")
    if not os.path.exists(ndptl_path):
        tar_path = os.path.join(workdir, f"{jobname}-{inst_id}.ensi.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:gz") as tar:
                member = [m for m in tar.getmembers() if 'NDPTL' in m.name][0]
                with tar.extractfile(member) as f:
                    lines = f.read().decode().splitlines()
        else:
            raise FileNotFoundError(f"No EnSight output for {jobname}-{inst_id}")
    else:
        with open(ndptl_path) as f:
            lines = f.readlines()

    values = [float(l.strip()) for l in lines[4:] if l.strip()]
    nodes, _, _, _ = parse_d_file(os.path.join(workdir, f"{jobname}.d"))
    coords = np.array(list(nodes.values()))
    z_min  = coords[:, 2].min()

    # First interior layer above z_min (downstream face, C=0)
    layer1_nodes = [nid for nid, xyz in nodes.items()
                    if abs(xyz[2] - (z_min + dz)) < dz * 0.05]
    if not layer1_nodes:
        layer1_nodes = [nid for nid, xyz in nodes.items()
                        if abs(xyz[2] - z_min) < dz * 0.05]

    avg_c = np.mean([values[nid - 1] for nid in layer1_nodes])
    # Multiply by D to convert concentration gradient [mol/m^4] to flux [mol/m^2/s]
    return D * avg_c / dz


def run_study():
    J_analytical = D * C_UP / T
    print(f"Analytical J = {J_analytical:.4e} mol/m^2/s")
    # n=10 and n=20 cause SIGABRT in rfemfield_thermal (LAS3D size constraint).
    # Use n=5, 8, 16, 32 — all confirmed working.
    print(f"Mesh sizes: n = 5, 8, 16, 32 (nxe = nye = n, nze = n)")

    results = []
    for n in [5, 8, 16, 32]:
        aa = L / n
        bb = L / n
        cc = T / n
        nels = n * n * n

        print(f"\nRunning n={n} ({nels} elements, element size {aa:.2e} x {bb:.2e} x {cc:.2e})...")

        with tempfile.TemporaryDirectory() as tmp:
            # Write rfem.rf directly (avoid printf escaping issues)
            write_rfem_rf(tmp, nels, n, aa, bb, cc, D, 1e-20)

            setup_script = (
                f"rfemcube rfem model\n"
                f"rfembc_thermal model {C_UP} 0.0\n"
                f"rfemfield_thermal rfem model 001\n"
                f"mpirun -np 1 rfemsolve_thermal model 001\n"
            )
            r = docker_run(tmp, IMAGE_RFEM, setup_script)
            if r.returncode != 0:
                print(f"  Pipeline failed for n={n}")
                continue

            try:
                J = extract_flux(tmp, "model", "001", cc)
                error_pct = abs(J - J_analytical) / J_analytical * 100
                print(f"  J = {J:.4e}, analytical = {J_analytical:.4e}, error = {error_pct:.2f}%")
                results.append({
                    'n': n, 'nels': nels, 'J': J,
                    'J_analytical': J_analytical, 'error_pct': error_pct,
                })
            except Exception as e:
                print(f"  Extraction failed for n={n}: {e}")

    df = pd.DataFrame(results)
    os.makedirs('studies/results', exist_ok=True)
    df.to_csv('studies/results/mesh_convergence.csv', index=False)
    print(f"\nResults:\n{df.to_string(index=False)}")

    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['n'], df['J'] / J_analytical, 'o-', label='RFEM Solver')
    plt.axhline(1.0, color='r', linestyle='--', label='Analytical')
    plt.ylim(0.99, 1.01)
    plt.xlabel('Elements per side (n)')
    plt.ylabel('J / J_analytical')
    plt.title('Mesh Convergence — Flux ratio')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(df['n'], df['error_pct'], 'o-')
    plt.axhline(1.0, color='r', linestyle='--', label='1% threshold')
    plt.xlabel('Elements per side (n)')
    plt.ylabel('Error vs analytical [%]')
    plt.title('Mesh Convergence — Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('studies/results/mesh_convergence.png', format='png')
    print("Saved studies/results/mesh_convergence.png")


if __name__ == '__main__':
    run_study()
