"""Integration tests for the full Galaxy tool pipeline.

Requires Docker images to be built locally with :local tags:
  parafem-meshgen:local
  parafem-bcgen:local
  parafem-p123:local
  parafem-vtu:local

Run with: pytest tests/integration/ -v --tb=short
Skip with: pytest tests/integration/ -k "not integration"
"""

import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TOOLS_DIR = os.path.join(REPO_ROOT, 'galaxy', 'tools')


def docker_available():
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def image_exists(name):
    result = subprocess.run(
        ['docker', 'image', 'inspect', name],
        capture_output=True, timeout=10
    )
    return result.returncode == 0


skip_no_docker = pytest.mark.skipif(
    not docker_available(), reason='Docker not available'
)


def run_in_container(image, workdir, command, timeout=60):
    result = subprocess.run(
        ['docker', 'run', '--rm',
         '-v', f'{workdir}:/work',
         '-v', f'{TOOLS_DIR}:/tools',
         image, 'bash', '-c', command],
        capture_output=True, text=True, timeout=timeout
    )
    return result


@skip_no_docker
class TestMeshGeneration:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem-meshgen:local'):
            pytest.skip('parafem-meshgen:local not built')

    def test_generates_mesh(self):
        cmd = """
NXE=2; NYE=2; NZE=2
NELS=$((NXE * NYE * NZE))
AA=$(awk "BEGIN {printf \\"%.8f\\", 1.0 / $NXE}")
BB=$(awk "BEGIN {printf \\"%.8f\\", 1.0 / $NYE}")
CC=$(awk "BEGIN {printf \\"%.8f\\", 1.0 / $NZE}")
cat > job.mg << MGEOF
'p123'
'parafem' ${NELS} ${NXE} ${NZE} 8
${AA} ${BB} ${CC} 1.0 1.0 1.0
1.0e-6 2000
0 0
MGEOF
p12meshgen job 2>&1
"""
        result = run_in_container('parafem-meshgen:local', self.workdir, cmd)
        assert result.returncode == 0, f"meshgen failed: {result.stderr}"
        assert os.path.exists(os.path.join(self.workdir, 'job.d'))
        assert os.path.exists(os.path.join(self.workdir, 'job.bnd'))

    def test_correct_node_count(self):
        self.test_generates_mesh()
        d_path = os.path.join(self.workdir, 'job.d')
        with open(d_path) as f:
            content = f.read()
        node_count = content.count('\n') - content.split('*ELEMENTS')[0].count('\n')
        # 2x2x2 hex = 27 nodes, rest are element lines
        assert '*NODES' in content
        assert '*ELEMENTS' in content


@skip_no_docker
class TestBCGenerator:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem-bcgen:local'):
            pytest.skip('parafem-bcgen:local not built')
        # Copy fixture mesh
        import shutil
        fixtures = os.path.join(REPO_ROOT, 'tests', 'fixtures')
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.d'), os.path.join(self.workdir, 'job.d'))

    def test_generates_bc_files(self):
        zone_json = '[{"axis_min": 0.0, "axis_max": 0.4, "temperature": 800.0}, {"axis_min": 0.6, "axis_max": 1.0, "temperature": 400.0}]'
        cmd = f"""python3 /tools/gdps_bc_thermal/gdps_bc_thermal.py \
            --mesh_d /work/job.d \
            --kx 50.0 --ky 50.0 --kz 50.0 \
            --tol 1.0e-6 --limit 2000 \
            --zone_config '{zone_json}' \
            --bc_axis z \
            --output_dat /work/job.dat \
            --output_bnd /work/job.bnd \
            --output_fix /work/job.fix \
            --output_d /work/job_out.d"""
        result = run_in_container('parafem-bcgen:local', self.workdir, cmd)
        assert result.returncode == 0, f"BC gen failed: {result.stderr}\n{result.stdout}"
        assert os.path.exists(os.path.join(self.workdir, 'job.dat'))
        assert os.path.exists(os.path.join(self.workdir, 'job.fix'))

    def test_dat_has_nr_zero(self):
        self.test_generates_bc_files()
        with open(os.path.join(self.workdir, 'job.dat')) as f:
            lines = f.readlines()
        parts = lines[3].split()
        nr = int(parts[2])
        assert nr == 0, f"nr should be 0, got {nr}"

    def test_fix_has_entries(self):
        self.test_generates_bc_files()
        with open(os.path.join(self.workdir, 'job.fix')) as f:
            lines = f.readlines()
        assert len(lines) == 18

    def test_bnd_is_empty(self):
        self.test_generates_bc_files()
        with open(os.path.join(self.workdir, 'job.bnd')) as f:
            content = f.read()
        assert content.strip() == ''


@skip_no_docker
class TestSolver:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem-p123:local'):
            pytest.skip('parafem-p123:local not built')
        # Copy all needed fixtures
        import shutil
        fixtures = os.path.join(REPO_ROOT, 'tests', 'fixtures')
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.d'), os.path.join(self.workdir, 'job.d'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.dat'), os.path.join(self.workdir, 'job.dat'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2_bc.bnd'), os.path.join(self.workdir, 'job.bnd'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.fix'), os.path.join(self.workdir, 'job.fix'))

    def _run_solver(self, np=1):
        cmd = f"mkdir -p run && cd run && cp /work/job.* . && mpirun -np {np} p123 job 2>&1 && cp job.res job.ensi.* /work/"
        return run_in_container('parafem-p123:local', self.workdir, cmd)

    def test_solver_runs(self):
        result = self._run_solver()
        assert result.returncode == 0, f"Solver failed: {result.stderr}\n{result.stdout}"
        assert os.path.exists(os.path.join(self.workdir, 'job.res'))

    def test_solver_converges(self):
        self._run_solver()
        with open(os.path.join(self.workdir, 'job.res')) as f:
            content = f.read()
        match = re.search(r'iterations to convergence was\s+(\d+)', content)
        assert match, "Could not find convergence info in .res"
        iters = int(match.group(1))
        assert iters < 100, f"Solver took {iters} iterations (should converge quickly)"

    def test_no_nan_in_results(self):
        self._run_solver()
        with open(os.path.join(self.workdir, 'job.res')) as f:
            content = f.read()
        assert 'NaN' not in content, "NaN found in results"

    def test_produces_ensi_output(self):
        self._run_solver()
        assert os.path.exists(os.path.join(self.workdir, 'job.ensi.NDPTL-000001'))

    def test_solver_with_2_procs(self):
        result = self._run_solver(np=2)
        assert result.returncode == 0
        with open(os.path.join(self.workdir, 'job.res')) as f:
            content = f.read()
        assert 'NaN' not in content


@skip_no_docker
class TestVTUConverter:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem-vtu:local'):
            pytest.skip('parafem-vtu:local not built')
        import shutil
        fixtures = os.path.join(REPO_ROOT, 'tests', 'fixtures')
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.d'), os.path.join(self.workdir, 'job.d'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.fix'), os.path.join(self.workdir, 'job.fix'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.ensi.NDPTL-000001'),
                    os.path.join(self.workdir, 'job.ensi.NDPTL-000001'))

    def test_mesh_only(self):
        cmd = "python3 /tools/gdps_parafem2vtu/parafem2vtu.py --mesh_d /work/job.d --output /work/result.vtu"
        result = run_in_container('parafem-vtu:local', self.workdir, cmd)
        assert result.returncode == 0, f"VTU failed: {result.stderr}\n{result.stdout}"
        vtu_path = os.path.join(self.workdir, 'result.vtu')
        assert os.path.exists(vtu_path)
        tree = ET.parse(vtu_path)
        piece = tree.find('.//Piece')
        assert piece.attrib['NumberOfPoints'] == '27'
        assert piece.attrib['NumberOfCells'] == '8'

    def test_with_solver_output(self):
        cmd = """python3 /tools/gdps_parafem2vtu/parafem2vtu.py \
            --mesh_d /work/job.d \
            --fix /work/job.fix \
            --ensi_dir /work \
            --jobname job \
            --output /work/result.vtu"""
        result = run_in_container('parafem-vtu:local', self.workdir, cmd)
        assert result.returncode == 0, f"VTU failed: {result.stderr}\n{result.stdout}"
        tree = ET.parse(os.path.join(self.workdir, 'result.vtu'))
        potential = tree.find('.//PointData/DataArray[@Name="Potential"]')
        assert potential is not None
        vals = [float(x) for x in potential.text.strip().split()]
        assert len(vals) == 27
        assert all(not (v != v) for v in vals), "NaN in VTU output"

    def test_vtu_is_valid_xml(self):
        cmd = "python3 /tools/gdps_parafem2vtu/parafem2vtu.py --mesh_d /work/job.d --output /work/result.vtu"
        run_in_container('parafem-vtu:local', self.workdir, cmd)
        tree = ET.parse(os.path.join(self.workdir, 'result.vtu'))
        root = tree.getroot()
        assert root.tag == 'VTKFile'
        assert root.attrib['type'] == 'UnstructuredGrid'


@skip_no_docker
class TestFullPipeline:
    """End-to-end test: meshgen -> BC gen -> solver -> VTU converter."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        for img in ['parafem-meshgen:local', 'parafem-bcgen:local',
                     'parafem-p123:local', 'parafem-vtu:local']:
            if not image_exists(img):
                pytest.skip(f'{img} not built')

    def test_end_to_end(self):
        # Step 1: Mesh generation
        mesh_cmd = """
NXE=2; NYE=2; NZE=2
NELS=$((NXE * NYE * NZE))
AA=$(awk "BEGIN {printf \\"%.8f\\", 1.0 / $NXE}")
BB=$(awk "BEGIN {printf \\"%.8f\\", 1.0 / $NYE}")
CC=$(awk "BEGIN {printf \\"%.8f\\", 1.0 / $NZE}")
cat > job.mg << MGEOF
'p123'
'parafem' ${NELS} ${NXE} ${NZE} 8
${AA} ${BB} ${CC} 1.0 1.0 1.0
1.0e-6 2000
0 0
MGEOF
p12meshgen job 2>&1
"""
        r = run_in_container('parafem-meshgen:local', self.workdir, mesh_cmd)
        assert r.returncode == 0, f"Meshgen failed: {r.stdout}\n{r.stderr}"

        # Step 2: BC generation
        zone_json = '[{"axis_min":0.0,"axis_max":0.4,"temperature":800.0},{"axis_min":0.6,"axis_max":1.0,"temperature":400.0}]'
        bc_cmd = f"""python3 /tools/gdps_bc_thermal/gdps_bc_thermal.py \
            --mesh_d /work/job.d --kx 50 --ky 50 --kz 50 \
            --tol 1e-6 --limit 2000 \
            --zone_config '{zone_json}' --bc_axis z \
            --output_dat /work/job.dat --output_bnd /work/job.bnd \
            --output_fix /work/job.fix --output_d /work/job_bc.d"""
        r = run_in_container('parafem-bcgen:local', self.workdir, bc_cmd)
        assert r.returncode == 0, f"BC gen failed: {r.stdout}\n{r.stderr}"

        # Step 3: Solver
        solver_cmd = "mkdir -p run && cd run && cp /work/job.dat /work/job.d /work/job.bnd /work/job.fix . && mpirun -np 1 p123 job 2>&1 && cp job.res job.ensi.* /work/"
        r = run_in_container('parafem-p123:local', self.workdir, solver_cmd)
        assert r.returncode == 0, f"Solver failed: {r.stdout}\n{r.stderr}"

        # Verify solver convergence
        with open(os.path.join(self.workdir, 'job.res')) as f:
            res = f.read()
        assert 'NaN' not in res
        match = re.search(r'iterations to convergence was\s+(\d+)', res)
        assert match and int(match.group(1)) < 100

        # Step 4: VTU conversion
        vtu_cmd = """python3 /tools/gdps_parafem2vtu/parafem2vtu.py \
            --mesh_d /work/job.d --fix /work/job.fix \
            --ensi_dir /work --jobname job --output /work/result.vtu"""
        r = run_in_container('parafem-vtu:local', self.workdir, vtu_cmd)
        assert r.returncode == 0, f"VTU failed: {r.stdout}\n{r.stderr}"

        # Verify VTU output
        tree = ET.parse(os.path.join(self.workdir, 'result.vtu'))
        piece = tree.find('.//Piece')
        assert piece.attrib['NumberOfPoints'] == '27'
        assert piece.attrib['NumberOfCells'] == '8'

        potential = tree.find('.//PointData/DataArray[@Name="Potential"]')
        assert potential is not None
        vals = [float(x) for x in potential.text.strip().split()]
        assert len(vals) == 27
        assert all(300 <= v <= 900 for v in vals)
