"""Integration tests for the Sprint 6c RFEM complex geometry mapping.

Tests the mapping of a structured RF field onto arbitrary meshes.
Requires: docker image williamjsmith15/parafem-rfem:local
"""

import os
import shutil
import subprocess
import tarfile
import tempfile
import pytest
import numpy as np
import re

IMAGE_RFEM   = 'williamjsmith15/parafem-rfem:local'
IMAGE_PYTHON = 'python:3.11-slim'

# Minimal .dat for a 1-element 8-node mesh (no BCs — used for mapping-only tests)
MINIMAL_DAT_1EL = """\
'hexahedron'
2
1
1
       1        8        0        8        8        0        0
  1.0E-06     2000   0.0E+00
"""

# Minimal .dat for a 1-element mesh with 2 BC nodes (upstream + downstream faces of a 1-elem box)
MINIMAL_DAT_1EL_2BC = """\
'hexahedron'
2
1
1
       1        8        0        8        8        0        2
  1.0E-06     2000   0.0E+00
"""


def docker_run(workdir, image, command):
    r = subprocess.run(
        ['docker', 'run', '--rm',
         '-v', f'{workdir}:/work', '-w', '/work',
         '-e', 'PYTHONDONTWRITEBYTECODE=1',
         image, 'bash', '-c', command],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        print(f"STDOUT: {r.stdout}")
        print(f"STDERR: {r.stderr}")
    return r


def copy_tools(workdir):
    os.makedirs(os.path.join(workdir, 'tools'), exist_ok=True)
    subprocess.run(['cp', 'galaxy/tools/gdps_rfem_map/gdps_rfem_map.py',
                    os.path.join(workdir, 'tools/')])
    subprocess.run(['cp', 'galaxy/tools/parafem_common.py', workdir])


@pytest.fixture
def workdir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


TARGET_D_1EL = """\
*THREE_DIMENSIONAL
*NODES
       1  0.0  0.0  0.0
       2  0.1  0.0  0.0
       3  0.1  0.1  0.0
       4  0.0  0.1  0.0
       5  0.0  0.0  0.1
       6  0.1  0.0  0.1
       7  0.1  0.1  0.1
       8  0.0  0.1  0.1
*ELEMENTS
       1 3 8 1 1 2 3 4 5 6 7 8 1
"""


def test_map_produces_mat_with_correct_nels(workdir):
    rf_script = """
cat > rfem.rf << 'EOF'
sim3de
1
8 2 2
0.1 0.1 0.1
0.5 0.5 0.5
1.0e-9 0.2e-9
dlavx3
0.3
EOF
rfemfield_thermal rfem
"""
    docker_run(workdir, IMAGE_RFEM, rf_script)
    assert os.path.exists(os.path.join(workdir, 'rfem.mat'))

    with open(os.path.join(workdir, 'target.d'), 'w') as f:
        f.write(TARGET_D_1EL)
    with open(os.path.join(workdir, 'target.dat'), 'w') as f:
        f.write(MINIMAL_DAT_1EL)

    copy_tools(workdir)

    map_script = """
pip install numpy > /dev/null 2>&1
python3 tools/gdps_rfem_map.py \
    --rf_mat rfem.mat \
    --target_d target.d \
    --dat_file target.dat \
    --instance_id 001 \
    --nxe 2 --nze 2 --aa 0.1 --bb 0.1 --cc 0.1 \
    --output_tar output.tar.gz \
    --output_res stats.txt
"""
    r = docker_run(workdir, IMAGE_PYTHON, map_script)
    assert r.returncode == 0, f"Map script failed: {r.stderr}"

    assert os.path.exists(os.path.join(workdir, 'output.tar.gz'))
    with tarfile.open(os.path.join(workdir, 'output.tar.gz')) as tar:
        tar.extractall(path=os.path.join(workdir, 'extracted'))

    mat_path = os.path.join(workdir, 'extracted', 'target-001.mat')
    with open(mat_path) as f:
        lines = f.readlines()
    assert lines[0].startswith('*MATERIAL 1 1')

    with open(os.path.join(workdir, 'stats.txt')) as f:
        stats = f.read()
    assert "Target elements: 1" in stats


def test_map_values_positive(workdir):
    """All mapped D values must be positive (log-normal field)."""
    rf_script = """
cat > rfem.rf << 'EOF'
sim3de
1
8 2 2
0.1 0.1 0.1
0.5 0.5 0.5
1.0e-9 0.3e-9
dlavx3
0.3
EOF
rfemfield_thermal rfem
"""
    docker_run(workdir, IMAGE_RFEM, rf_script)

    # 8-element target mesh (2x2x2 box matching the RF grid)
    with open(os.path.join(workdir, 'target.d'), 'w') as f:
        f.write(TARGET_D_1EL)
    with open(os.path.join(workdir, 'target.dat'), 'w') as f:
        f.write(MINIMAL_DAT_1EL)

    copy_tools(workdir)

    map_script = """
pip install numpy > /dev/null 2>&1
python3 tools/gdps_rfem_map.py \
    --rf_mat rfem.mat \
    --target_d target.d \
    --dat_file target.dat \
    --nxe 2 --nze 2 --aa 0.1 --bb 0.1 --cc 0.1 \
    --output_tar output.tar.gz \
    --output_res stats.txt
"""
    r = docker_run(workdir, IMAGE_PYTHON, map_script)
    assert r.returncode == 0, f"Map script failed: {r.stderr}"

    with tarfile.open(os.path.join(workdir, 'output.tar.gz')) as tar:
        tar.extractall(path=os.path.join(workdir, 'extracted'))

    mat_path = None
    for name in os.listdir(os.path.join(workdir, 'extracted')):
        if name.endswith('.mat'):
            mat_path = os.path.join(workdir, 'extracted', name)
    assert mat_path is not None

    d_vals = []
    with open(mat_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    d_vals.append(float(parts[1]))
                except ValueError:
                    pass
    assert all(v > 0 for v in d_vals), f"Non-positive D values found: {[v for v in d_vals if v <= 0]}"


def test_map_preserves_mean(workdir):
    rf_script = """
cat > rfem.rf << 'EOF'
sim3de
1
64 4 4
0.1 0.1 0.1
1.0 1.0 1.0
1.0e-9 1e-10
dlavx3
0.3
EOF
rfemfield_thermal rfem
"""
    docker_run(workdir, IMAGE_RFEM, rf_script)

    target_d = """\
*THREE_DIMENSIONAL
*NODES
       1  0.05  0.05  0.05
       2  0.15  0.05  0.05
       3  0.15  0.15  0.05
       4  0.05  0.15  0.05
       5  0.05  0.05  0.15
       6  0.15  0.05  0.15
       7  0.15  0.15  0.15
       8  0.05  0.15  0.15
*ELEMENTS
       1 3 8 1 1 2 3 4 5 6 7 8 1
"""
    with open(os.path.join(workdir, 'target.d'), 'w') as f:
        f.write(target_d)
    with open(os.path.join(workdir, 'target.dat'), 'w') as f:
        f.write(MINIMAL_DAT_1EL)

    copy_tools(workdir)

    map_script = """
pip install numpy > /dev/null 2>&1
python3 tools/gdps_rfem_map.py \
    --rf_mat rfem.mat \
    --target_d target.d \
    --dat_file target.dat \
    --nxe 4 --nze 4 --aa 0.1 --bb 0.1 --cc 0.1 \
    --output_tar output.tar.gz \
    --output_res stats.txt
"""
    r = docker_run(workdir, IMAGE_PYTHON, map_script)
    assert r.returncode == 0, f"Map script failed: {r.stderr}"

    with open(os.path.join(workdir, 'stats.txt')) as f:
        stats = f.read()
    mean_match = re.search(r'Mapped D mean: ([\d.e+-]+)', stats)
    assert mean_match
    mean_val = float(mean_match.group(1))
    assert 5e-10 < mean_val < 2e-9


def test_full_pipeline_with_map(workdir):
    """map → solve → verify fixed_freedoms preserved and solution physically reasonable.

    Uses rfemcube (2x2x2 = 8 elements, 27 nodes) so interior nodes exist for PCG.
    """
    # Use rfemcube to generate a proper structured mesh
    setup_script = """\
cat > rfem.rf << 'EOF'
sim3de
1
8 2 2
0.1 0.1 0.1
0.5 0.5 0.5
50.0 5.0
dlavx3
0.3
EOF
rfemcube rfem model
rfembc_thermal model 100.0 20.0
rfemfield_thermal rfem
"""
    r = docker_run(workdir, IMAGE_RFEM, setup_script)
    assert r.returncode == 0, r.stderr

    copy_tools(workdir)
    map_script = """\
pip install numpy > /dev/null 2>&1
python3 tools/gdps_rfem_map.py \
    --rf_mat rfem.mat --target_d model.d --dat_file model.dat --instance_id 001 \
    --nxe 2 --nze 2 --aa 0.1 --bb 0.1 --cc 0.1 \
    --output_tar model-001.tar.gz --output_res stats.txt
"""
    r = docker_run(workdir, IMAGE_PYTHON, map_script)
    assert r.returncode == 0, f"Map script failed: {r.stderr}"

    # Verify fixed_freedoms was preserved in the archive's .dat
    with tarfile.open(os.path.join(workdir, 'model-001.tar.gz')) as tar:
        tar.extractall(path=os.path.join(workdir, 'extracted'))
    dat_path = os.path.join(workdir, 'extracted', 'model-001.dat')
    with open(dat_path) as f:
        dat_tokens = f.read().split()
    fixed_freedoms = int(dat_tokens[10])
    assert fixed_freedoms > 0, f"fixed_freedoms should be >0 in mapped .dat, got {fixed_freedoms}"

    solve_script = """\
cp model.fix model-001.fix
tar xzf model-001.tar.gz
mpirun -np 1 rfemsolve_thermal model 001
"""
    r = docker_run(workdir, IMAGE_RFEM, solve_script)
    assert r.returncode == 0, f"Solve failed: {r.stderr}"
    assert os.path.exists(os.path.join(workdir, 'model-001.res'))

    # Verify solution is physically reasonable: values between 20 and 100
    ensi_files = [f for f in os.listdir(workdir) if 'NDPTL' in f]
    if ensi_files:
        with open(os.path.join(workdir, ensi_files[0])) as f:
            lines = f.readlines()
        values = [float(l.strip()) for l in lines[4:] if l.strip()]
        finite_vals = [v for v in values if not (v != v)]  # exclude NaN
        assert len(finite_vals) > 0, "No finite values in solution"
        assert min(finite_vals) >= 15.0, f"Min value too low: {min(finite_vals)}"
        assert max(finite_vals) <= 105.0, f"Max value too high: {max(finite_vals)}"
        assert max(finite_vals) > min(finite_vals), "Solution is uniform — BCs may not be applied"
