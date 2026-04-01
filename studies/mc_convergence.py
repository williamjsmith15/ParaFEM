"""RFEM MC Convergence Study.

Verifies that the variance of the estimated mean flux decreases at the 1/N
rate as the number of MC instances increases.

Usage:
    python studies/mc_convergence.py
Outputs (studies/results/):
    mc_convergence.csv
    mc_convergence.png
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

# GDPS membrane parameters (8x8x8 mesh — N_MESH=10 causes SIGABRT in rfemfield_thermal)
N_MESH   = 8
L        = 5e-3
T        = 0.5e-3
D_MEAN   = 1.5e-12
D_STD    = 0.3e-12   # 20% relative SD
C_UP     = 4.5e-3
N_VALUES = [50, 100, 200]


def docker_run(workdir, image, command, env=None):
    cmd = ['docker', 'run', '--rm', '-v', f'{workdir}:/work', '-w', '/work']
    for k, v in (env or {}).items():
        cmd += ['-e', f'{k}={v}']
    cmd += [image, 'bash', '-c', command]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDOUT: {r.stdout}")
        print(f"STDERR: {r.stderr}")
    return r


def extract_flux_proxy(workdir, jobname, inst_id, dz):
    """Extract flux proxy from EnSight NDPTL output."""
    ndptl_path = os.path.join(workdir, f"{jobname}-{inst_id}.ensi.NDPTL-000001")
    if not os.path.exists(ndptl_path):
        tar_path = os.path.join(workdir, f"{jobname}-{inst_id}.ensi.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:gz") as tar:
                member = [m for m in tar.getmembers() if 'NDPTL' in m.name][0]
                with tar.extractfile(member) as f:
                    lines = f.read().decode().splitlines()
        else:
            return None
    else:
        with open(ndptl_path) as f:
            lines = f.readlines()

    values = [float(l.strip()) for l in lines[4:] if l.strip()]
    nodes, _, _, _ = parse_d_file(os.path.join(workdir, f"{jobname}.d"))
    coords = np.array(list(nodes.values()))
    z_min  = coords[:, 2].min()
    cc     = (coords[:, 2].max() - z_min) / N_MESH

    layer1_nodes = [nid for nid, xyz in nodes.items()
                    if abs(xyz[2] - (z_min + cc)) < cc * 0.01]
    if not layer1_nodes:
        layer1_nodes = [nid for nid, xyz in nodes.items()
                        if abs(xyz[2] - z_min) < cc * 0.01]
    if not layer1_nodes:
        return None

    return np.mean([values[nid - 1] for nid in layer1_nodes]) / cc


def run_study():
    n    = N_MESH
    aa   = L / n
    bb   = L / n
    cc   = T / n
    nels = n * n * n

    J_analytical = D_MEAN * C_UP / T
    print(f"Analytical J (mean D) = {J_analytical:.4e} mol/m^2/s")
    print(f"Mesh: {n}x{n}x{n} = {nels} elements")
    print(f"D_mean = {D_MEAN:.2e}, D_std = {D_STD:.2e} (CoV = {D_STD/D_MEAN:.0%})")

    all_J = []
    max_N = max(N_VALUES)

    thx = 2 * aa
    thy = 2 * bb
    thz = 2 * cc
    rf_conf = (
        f"sim3de\n1\n{nels} {n} {n}\n{aa} {bb} {cc}\n"
        f"{thx} {thy} {thz}\n{D_MEAN} {D_STD}\ndlavx3\n0.3\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        # Write rfem.rf directly (avoid printf escaping issues)
        with open(os.path.join(tmp, 'rfem.rf'), 'w') as f:
            f.write(rf_conf)

        setup_script = (
            f"rfemcube rfem model\n"
            f"rfembc_thermal model {C_UP} 0.0\n"
        )
        r = docker_run(tmp, IMAGE_RFEM, setup_script)
        if r.returncode != 0:
            print("Setup failed — aborting study")
            return

        print(f"\nGenerating and solving {max_N} instances...")
        for i in range(1, max_N + 1):
            inst_id = f"{i:03d}"
            # Write fresh rfem.rf for each instance (rfemfield reads it)
            with open(os.path.join(tmp, 'rfem.rf'), 'w') as f:
                f.write(rf_conf)
            solve_script = (
                f"rfemfield_thermal rfem model {inst_id} > /dev/null 2>&1 && "
                f"cp model.fix model-{inst_id}.fix && "
                f"mpirun -np 1 rfemsolve_thermal model {inst_id} > /dev/null 2>&1"
            )
            docker_run(tmp, IMAGE_RFEM, solve_script, env={'RFEM_SEED': str(i)})
            J = extract_flux_proxy(tmp, "model", inst_id, cc)
            if J is not None:
                all_J.append(J)
            if i % 50 == 0:
                print(f"  Completed {i}/{max_N}, running mean = {np.mean(all_J):.4e}")

    print(f"\nCompleted {len(all_J)} instances successfully")

    results = []
    for N in N_VALUES:
        if N > len(all_J):
            continue
        subset      = all_J[:N]
        mean_J      = np.mean(subset)
        std_J       = np.std(subset, ddof=1)
        var_of_mean = (std_J ** 2) / N
        stderr      = std_J / np.sqrt(N)
        cov         = std_J / mean_J if mean_J != 0 else 0.0
        results.append({
            'N': N, 'mean_J': mean_J, 'std_J': std_J,
            'stderr': stderr, 'var_of_mean': var_of_mean, 'cov': cov,
        })
        print(f"  N={N:4d}: mean={mean_J:.4e}, std={std_J:.4e}, "
              f"stderr={stderr:.4e}, CoV={cov:.3f}")

    df = pd.DataFrame(results)
    os.makedirs('studies/results', exist_ok=True)
    df.to_csv('studies/results/mc_convergence.csv', index=False)

    # Log-log slope
    log_N   = np.log(df['N'])
    log_var = np.log(df['var_of_mean'])
    slope, _ = np.polyfit(log_N, log_var, 1)
    print(f"\nLog-log slope of var(mean_J) vs N: {slope:.4f} (expected -1.0)")

    with open('studies/results/mc_slope.txt', 'w') as f:
        f.write(f"{slope:.4f}\n")

    plt.figure(figsize=(8, 5))
    plt.loglog(df['N'], df['var_of_mean'], 'o-', label='Observed variance of mean')
    C_fit = df['var_of_mean'].iloc[0] * df['N'].iloc[0]
    plt.loglog(df['N'], C_fit / df['N'], 'r--', label='Theoretical 1/N')
    plt.xlabel('Number of MC instances (N)')
    plt.ylabel('Variance of mean flux')
    plt.title('MC Convergence Study')
    plt.legend()
    plt.grid(True, which='both')
    plt.savefig('studies/results/mc_convergence.png')
    print("Saved studies/results/mc_convergence.png")

    return slope


if __name__ == '__main__':
    run_study()
