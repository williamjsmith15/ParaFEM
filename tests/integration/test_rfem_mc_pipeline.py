"""Integration tests for the Sprint 6b RFEM MC pipeline.

Tests the batch runner, aggregator, and full MC pipeline.
Requires: docker image williamjsmith15/parafem-rfem:local
"""

import os
import shutil
import subprocess
import tarfile
import tempfile
import pytest
import json
import pandas as pd

IMAGE_RFEM   = 'williamjsmith15/parafem-rfem:local'
IMAGE_PYTHON = 'python:3.11-slim'

RF_CONF_2x2x2 = """\
sim3de
2
8 2 2
0.1 0.1 0.1
0.5 0.5 0.5
1.0e-9 0.2e-9
dlavx3
0.3
"""


def docker_run(workdir, image, command):
    return subprocess.run(
        ['docker', 'run', '--rm',
         '-v', f'{workdir}:/work', '-w', '/work',
         '-e', 'PYTHONDONTWRITEBYTECODE=1',
         image, 'bash', '-c', command],
        capture_output=True, text=True, timeout=180
    )


@pytest.fixture
def workdir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_mesh(workdir):
    """Generate mesh + BCs for 2x2x2 RFEM model."""
    script = (
        "cat > rfem.rf << 'EOF'\n"
        + RF_CONF_2x2x2
        + "EOF\n"
        "rfemcube rfem model\n"
        "rfembc_thermal model 1.0 0.0\n"
    )
    r = docker_run(workdir, IMAGE_RFEM, script)
    assert r.returncode == 0, f"Setup failed: {r.stderr}"


def _run_n_instances(workdir, n):
    """Generate + solve N RFEM instances. Files land in workdir/outputs/."""
    script = (
        "set -e\n"
        "mkdir -p outputs\n"
        f"for i in $(seq -f '%03g' 1 {n}); do\n"
        "  printf 'sim3de\\n2\\n8 2 2\\n0.1 0.1 0.1\\n0.5 0.5 0.5\\n1.0e-9 0.2e-9\\ndlavx3\\n0.3\\n' > rfem.rf\n"
        "  rfemfield_thermal rfem model ${i}\n"
        "  cp model.fix model-${i}.fix\n"
        "  mpirun -np 1 rfemsolve_thermal model ${i}\n"
        "  cp model-${i}.res outputs/model-${i}.res\n"
        "  tar czf outputs/model-${i}.ensi.tar.gz model-${i}.ensi.*\n"
        "done\n"
    )
    r = docker_run(workdir, IMAGE_RFEM, script)
    assert r.returncode == 0, f"MC run failed: {r.stderr}"


def _copy_collect_tool(workdir):
    os.makedirs(os.path.join(workdir, 'tools'), exist_ok=True)
    subprocess.run(['cp', 'galaxy/tools/gdps_rfemmc_collect/gdps_rfemmc_collect.py',
                    os.path.join(workdir, 'tools/')], check=True)
    subprocess.run(['cp', 'galaxy/tools/parafem_common.py', workdir], check=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mc_runner_produces_n_archives(workdir):
    _setup_mesh(workdir)

    script = (
        "mkdir -p run outputs && cd run && cp ../model.d model.d && cp ../model.dat model.dat &&\n"
        "for i in $(seq -f '%03g' 1 3); do\n"
        "  printf 'sim3de\\n1\\n8 2 2\\n0.1 0.1 0.1\\n0.5 0.5 0.5\\n1.0e-9 0.2e-9\\ndlavx3\\n0.3\\n' > rfem.rf &&\n"
        "  rfemfield_thermal rfem model ${i} > /dev/null 2>&1 &&\n"
        "  tar czf \"../outputs/model-${i}.tar.gz\" \"model-${i}.d\" \"model-${i}.dat\" \"model-${i}.mat\";\n"
        "done\n"
    )
    r = docker_run(workdir, IMAGE_RFEM, script)
    assert r.returncode == 0, r.stderr

    outputs_dir = os.path.join(workdir, 'outputs')
    archives    = sorted(os.listdir(outputs_dir))
    assert archives == ['model-001.tar.gz', 'model-002.tar.gz', 'model-003.tar.gz']

    with tarfile.open(os.path.join(outputs_dir, 'model-001.tar.gz')) as tar:
        names = tar.getnames()
        assert 'model-001.d'   in names
        assert 'model-001.dat' in names
        assert 'model-001.mat' in names


def test_instances_differ(workdir):
    _setup_mesh(workdir)

    script = (
        "mkdir -p run && cd run && cp ../model.d model.d && cp ../model.dat model.dat &&\n"
        "for i in $(seq -f '%03g' 1 2); do\n"
        "  printf 'sim3de\\n1\\n8 2 2\\n0.1 0.1 0.1\\n0.5 0.5 0.5\\n1.0e-9 0.2e-9\\ndlavx3\\n0.3\\n' > rfem.rf &&\n"
        "  rfemfield_thermal rfem model ${i} > /dev/null 2>&1;\n"
        "done\n"
    )
    docker_run(workdir, IMAGE_RFEM, script)

    def read_mat(path):
        with open(path) as f:
            return f.read()

    mat1 = read_mat(os.path.join(workdir, 'run', 'model-001.mat'))
    mat2 = read_mat(os.path.join(workdir, 'run', 'model-002.mat'))
    assert mat1 != mat2


def test_collect_output_format(workdir):
    # Generate a real .d with rfemcube, then use dummy ensi data
    docker_run(workdir, IMAGE_RFEM,
               "cat > rfem.rf << 'EOF'\n" + RF_CONF_2x2x2 + "EOF\nrfemcube rfem model\n")

    os.makedirs(os.path.join(workdir, 'inputs'), exist_ok=True)

    with open(os.path.join(workdir, 'inputs', 'res-001.res'), 'w') as f:
        f.write("The number of iterations to convergence was 42\n")

    ndptl_content = "EnSight Gold Scalar\npart\n1\ncoordinates\n" + "\n".join(["0.5"] * 27) + "\n"
    with open(os.path.join(workdir, 'inputs', 'model-001.ensi.NDPTL-000001'), 'w') as f:
        f.write(ndptl_content)

    with tarfile.open(os.path.join(workdir, 'inputs', 'ensi-001.tar.gz'), 'w:gz') as tar:
        tar.add(os.path.join(workdir, 'inputs', 'model-001.ensi.NDPTL-000001'),
                arcname='model-001.ensi.NDPTL-000001')

    _copy_collect_tool(workdir)

    script = (
        "pip install numpy pandas matplotlib > /dev/null 2>&1\n"
        "python tools/gdps_rfemmc_collect.py"
        " --d_file model.d"
        " --res_file 001 inputs/res-001.res"
        " --ensi_file 001 inputs/ensi-001.tar.gz"
        " --output_csv summary.csv"
        " --output_json stats.json"
        " --output_png hist.png\n"
    )
    r = docker_run(workdir, IMAGE_PYTHON, script)
    assert r.returncode == 0, r.stderr

    assert os.path.exists(os.path.join(workdir, 'summary.csv'))
    assert os.path.exists(os.path.join(workdir, 'stats.json'))
    assert os.path.exists(os.path.join(workdir, 'hist.png'))

    with open(os.path.join(workdir, 'stats.json')) as f:
        stats = json.load(f)
    assert stats['count'] == 1
    assert stats['mean'] > 0

    df = pd.read_csv(os.path.join(workdir, 'summary.csv'))
    assert df.iloc[0]['iterations'] == 42


def test_flux_extraction_positive(workdir):
    """Full solve of 2 instances; extracted flux values must be positive."""
    _setup_mesh(workdir)
    _run_n_instances(workdir, 2)

    _copy_collect_tool(workdir)

    script = (
        "pip install numpy pandas matplotlib > /dev/null 2>&1\n"
        "python tools/gdps_rfemmc_collect.py"
        " --d_file model.d"
        " --ensi_file 001 outputs/model-001.ensi.tar.gz"
        " --ensi_file 002 outputs/model-002.ensi.tar.gz"
        " --output_csv summary.csv"
        " --output_json stats.json"
        " --output_png hist.png\n"
    )
    r = docker_run(workdir, IMAGE_PYTHON, script)
    assert r.returncode == 0, r.stderr

    with open(os.path.join(workdir, 'stats.json')) as f:
        stats = json.load(f)
    assert stats['count'] == 2
    assert stats['mean'] > 0, f"Mean flux should be positive, got {stats['mean']}"
    assert stats['min']  > 0, f"All flux values should be positive, got min={stats['min']}"


def test_full_mc_pipeline(workdir):
    """3-instance end-to-end: mesh → BCs → field → solve → collect."""
    _setup_mesh(workdir)
    _run_n_instances(workdir, 3)

    outputs = os.path.join(workdir, 'outputs')
    for i in ['001', '002', '003']:
        assert os.path.exists(os.path.join(outputs, f'model-{i}.res')), \
            f"Missing .res for instance {i}"

    _copy_collect_tool(workdir)

    script = (
        "pip install numpy pandas matplotlib > /dev/null 2>&1\n"
        "python tools/gdps_rfemmc_collect.py"
        " --d_file model.d"
        " --res_file  001 outputs/model-001.res"
        " --res_file  002 outputs/model-002.res"
        " --res_file  003 outputs/model-003.res"
        " --ensi_file 001 outputs/model-001.ensi.tar.gz"
        " --ensi_file 002 outputs/model-002.ensi.tar.gz"
        " --ensi_file 003 outputs/model-003.ensi.tar.gz"
        " --output_csv summary.csv"
        " --output_json stats.json"
        " --output_png hist.png\n"
    )
    r = docker_run(workdir, IMAGE_PYTHON, script)
    assert r.returncode == 0, r.stderr

    with open(os.path.join(workdir, 'stats.json')) as f:
        stats = json.load(f)
    assert stats['count'] == 3
    assert stats['mean'] > 0

    df = pd.read_csv(os.path.join(workdir, 'summary.csv'))
    assert len(df) == 3
