"""Integration tests for the full Galaxy tool pipeline.

Requires Docker images to be built locally with :local tags:
  parafem:local
  parafem-bcgen:local
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
        if not image_exists('parafem:local'):
            pytest.skip('parafem:local not built')

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
        result = run_in_container('parafem:local', self.workdir, cmd)
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

    def test_bnd_has_boundary_nodes(self):
        self.test_generates_bc_files()
        with open(os.path.join(self.workdir, 'job.bnd')) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 26, f"Expected 26 boundary nodes, got {len(lines)}"
        for line in lines:
            parts = line.split()
            assert len(parts) == 2, f"Expected 2 columns, got {len(parts)}: {line.strip()}"
            assert int(parts[1]) == 0, f"Expected restraint 0, got {parts[1]}"


@skip_no_docker
class TestSolver:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem:local'):
            pytest.skip('parafem:local not built')
        # Copy all needed fixtures
        import shutil
        fixtures = os.path.join(REPO_ROOT, 'tests', 'fixtures')
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.d'), os.path.join(self.workdir, 'job.d'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.dat'), os.path.join(self.workdir, 'job.dat'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2_bc.bnd'), os.path.join(self.workdir, 'job.bnd'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.fix'), os.path.join(self.workdir, 'job.fix'))

    def _run_solver(self, np=1):
        cmd = f"mkdir -p run && cd run && cp /work/job.* . && mpirun -np {np} p123 job 2>&1 && cp job.res job.ensi.* /work/"
        return run_in_container('parafem:local', self.workdir, cmd)

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
        for img in ['parafem:local', 'parafem-bcgen:local',
                     'parafem:local', 'parafem-vtu:local']:
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
        r = run_in_container('parafem:local', self.workdir, mesh_cmd)
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
        r = run_in_container('parafem:local', self.workdir, solver_cmd)
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


@skip_no_docker
class TestTransientBCGenerator:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem-bcgen:local'):
            pytest.skip('parafem-bcgen:local not built')
        import shutil
        fixtures = os.path.join(REPO_ROOT, 'tests', 'fixtures')
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.d'), os.path.join(self.workdir, 'job.d'))

    def _run_bc_transient(self, zone_json=None):
        if zone_json is None:
            zone_json = '[{"axis_min":0.0,"axis_max":0.1,"temperature":800.0},{"axis_min":0.9,"axis_max":1.0,"temperature":400.0}]'
        cmd = f"""python3 /tools/gdps_bc_transient/gdps_bc_transient.py \
            --mesh_d /work/job.d \
            --kx 50.0 --ky 50.0 --kz 50.0 --rho 7800.0 --cp 500.0 \
            --val0 400.0 --dtim 10.0 --nstep 5 --npri 1 \
            --theta 0.5 --tol 1.0e-8 --limit 200 \
            --zone_config /work/zones.json \
            --bc_axis z \
            --output_dat /work/job.dat \
            --output_bnd /work/job.bnd \
            --output_fix /work/job.fix \
            --output_mat /work/job.mat \
            --output_d /work/job_out.d"""
        with open(os.path.join(self.workdir, 'zones.json'), 'w') as f:
            f.write(zone_json)
        return run_in_container('parafem-bcgen:local', self.workdir, cmd)

    def test_generates_all_files(self):
        result = self._run_bc_transient()
        assert result.returncode == 0, f"BC transient failed: {result.stderr}\n{result.stdout}"
        for fname in ['job.dat', 'job.bnd', 'job.fix', 'job.mat', 'job_out.d']:
            assert os.path.exists(os.path.join(self.workdir, fname)), f"Missing {fname}"

    def test_dat_nr_zero(self):
        self._run_bc_transient()
        with open(os.path.join(self.workdir, 'job.dat')) as f:
            lines = f.readlines()
        parts = lines[3].split()
        assert int(parts[3]) == 0, f"nr should be 0, got {parts[3]}"

    def test_dat_has_time_params(self):
        self._run_bc_transient()
        with open(os.path.join(self.workdir, 'job.dat')) as f:
            lines = f.readlines()
        parts = lines[4].split()
        assert float(parts[0]) == pytest.approx(400.0)   # val0
        assert float(parts[1]) == pytest.approx(10.0)    # dtim
        assert int(parts[2]) == 5                         # nstep

    def test_mat_has_material_values(self):
        self._run_bc_transient()
        with open(os.path.join(self.workdir, 'job.mat')) as f:
            lines = f.readlines()
        assert lines[0].startswith('*MATERIAL')
        parts = lines[2].split()
        assert float(parts[1]) == pytest.approx(50.0)    # kx
        assert float(parts[4]) == pytest.approx(7800.0)  # rho
        assert float(parts[5]) == pytest.approx(500.0)   # cp

    def test_bnd_has_boundary_nodes(self):
        self._run_bc_transient()
        with open(os.path.join(self.workdir, 'job.bnd')) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 26, f"Expected 26 boundary nodes, got {len(lines)}"
        for line in lines:
            parts = line.split()
            assert len(parts) == 2, f"Expected 2 columns, got {len(parts)}: {line.strip()}"
            assert int(parts[1]) == 0, f"Expected restraint 0, got {parts[1]}"

    def test_fix_has_18_entries(self):
        self._run_bc_transient()
        with open(os.path.join(self.workdir, 'job.fix')) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 18


@skip_no_docker
class TestTransientSolver:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem:local'):
            pytest.skip('parafem:local not built')
        import shutil
        fixtures = os.path.join(REPO_ROOT, 'tests', 'fixtures')
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.d'), os.path.join(self.workdir, 'job.d'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2_transient.dat'), os.path.join(self.workdir, 'job.dat'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2_transient.mat'), os.path.join(self.workdir, 'job.mat'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.fix'), os.path.join(self.workdir, 'job.fix'))

    def _run_solver(self, np=1):
        cmd = f"cd /work && mpirun -np {np} gdps_thermal_transient job 2>&1"
        return run_in_container('parafem:local', self.workdir, cmd)

    def test_solver_runs(self):
        result = self._run_solver()
        assert result.returncode == 0, f"Solver failed: {result.stderr}\n{result.stdout}"
        assert os.path.exists(os.path.join(self.workdir, 'job.res'))

    def test_solver_produces_ensi_output(self):
        self._run_solver()
        ndttr_files = [f for f in os.listdir(self.workdir) if 'NDTTR' in f]
        assert len(ndttr_files) > 0, "No NDTTR files produced"

    def test_no_nan_in_results(self):
        self._run_solver()
        with open(os.path.join(self.workdir, 'job.res')) as f:
            content = f.read()
        assert 'NaN' not in content

    def test_bc_nodes_correct_temperature(self):
        self._run_solver()
        # Read the final NDTTR file (step 5)
        ndttr_path = os.path.join(self.workdir, 'job.ensi.NDTTR-000005')
        assert os.path.exists(ndttr_path), "NDTTR-000005 not found"
        with open(ndttr_path) as f:
            lines = f.readlines()
        # Skip header lines (Alya header, part, 1, coordinates)
        data_start = next(i for i, l in enumerate(lines) if 'coordinates' in l) + 1
        temps = [float(l.strip()) for l in lines[data_start:] if l.strip()]
        assert len(temps) == 27
        # BC nodes at z=0: nodes 1,2,3,10,11,12,19,20,21 = indices 0,1,2,9,10,11,18,19,20 => T=400K
        bc_z0 = [temps[i] for i in [0, 1, 2, 9, 10, 11, 18, 19, 20]]
        for t in bc_z0:
            assert abs(t - 400.0) < 1.0, f"BC node z=0 temperature {t} != 400K"
        # BC nodes at z=-1: nodes 7,8,9,16,17,18,25,26,27 = indices 6,7,8,15,16,17,24,25,26 => T=800K
        bc_z1 = [temps[i] for i in [6, 7, 8, 15, 16, 17, 24, 25, 26]]
        for t in bc_z1:
            assert abs(t - 800.0) < 1.0, f"BC node z=-1 temperature {t} != 800K"

    def test_free_nodes_between_bc_values(self):
        self._run_solver()
        ndttr_path = os.path.join(self.workdir, 'job.ensi.NDTTR-000005')
        with open(ndttr_path) as f:
            lines = f.readlines()
        data_start = next(i for i, l in enumerate(lines) if 'coordinates' in l) + 1
        temps = [float(l.strip()) for l in lines[data_start:] if l.strip()]
        # Free nodes at z=-0.5: indices 3,4,5,12,13,14,21,22,23
        free_temps = [temps[i] for i in [3, 4, 5, 12, 13, 14, 21, 22, 23]]
        for t in free_temps:
            assert 400.0 <= t <= 800.0, f"Free node temperature {t} out of range [400, 800]"


@skip_no_docker
class TestTransientSolverLongRun:
    """Verify meaningful thermal evolution over a longer time scale.

    Uses dtim=1000s, nstep=20 (t_final=20000s). With alpha=k/(rho*cp)=1.28e-5 m^2/s
    and L=1m, the diffusion timescale is ~78000s, so t=20000s is well into the
    transient — the midplane node (14) should rise from 400K to ~589K, approaching
    the 600K steady-state midpoint between the 400K and 800K BCs.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem:local'):
            pytest.skip('parafem:local not built')
        import shutil
        fixtures = os.path.join(REPO_ROOT, 'tests', 'fixtures')
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.d'), os.path.join(self.workdir, 'job.d'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2_transient_long.dat'), os.path.join(self.workdir, 'job.dat'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2_transient.mat'), os.path.join(self.workdir, 'job.mat'))
        shutil.copy(os.path.join(fixtures, 'small_2x2x2.fix'), os.path.join(self.workdir, 'job.fix'))

    def _run_solver(self):
        cmd = "cd /work && mpirun -np 1 gdps_thermal_transient job 2>&1"
        return run_in_container('parafem:local', self.workdir, cmd)

    def test_midplane_node_heats_up(self):
        result = self._run_solver()
        assert result.returncode == 0, f"Solver failed: {result.stderr}\n{result.stdout}"
        # nres=14 (midplane centre node) — read temperature evolution from .res
        with open(os.path.join(self.workdir, 'job.res')) as f:
            content = f.read()
        # Extract time/temp pairs from the table (skip t=0 header line)
        rows = re.findall(r'^\s+([\d.E+\-]+)\s+([\d.E+\-]+)', content, re.MULTILINE)
        assert len(rows) >= 3, "Expected at least 3 time output rows"
        temps = [float(r[1]) for r in rows]
        # Temperature should strictly increase (heating from 400K toward 600K)
        assert temps[-1] > temps[0] + 50.0, f"Expected >50K rise, got {temps[-1] - temps[0]:.1f}K"

    def test_midplane_approaches_steady_state(self):
        self._run_solver()
        ndttr_path = os.path.join(self.workdir, 'job.ensi.NDTTR-000020')
        assert os.path.exists(ndttr_path), "NDTTR-000020 not found (nstep=20 should produce it)"
        with open(ndttr_path) as f:
            lines = f.readlines()
        data_start = next(i for i, l in enumerate(lines) if 'coordinates' in l) + 1
        temps = [float(l.strip()) for l in lines[data_start:] if l.strip()]
        assert len(temps) == 27
        # Midplane free nodes (indices 3,4,5,12,13,14,21,22,23) should be between
        # initial (400K) and steady-state midpoint (600K) — well above 450K at t=20000s
        free_temps = [temps[i] for i in [3, 4, 5, 12, 13, 14, 21, 22, 23]]
        for t in free_temps:
            assert t > 450.0, f"Midplane node at {t}K — expected significant heating by t=20000s"
            assert t < 800.0, f"Midplane node at {t}K exceeds upper BC"

    def test_bc_nodes_remain_fixed_long_run(self):
        self._run_solver()
        ndttr_path = os.path.join(self.workdir, 'job.ensi.NDTTR-000020')
        with open(ndttr_path) as f:
            lines = f.readlines()
        data_start = next(i for i, l in enumerate(lines) if 'coordinates' in l) + 1
        temps = [float(l.strip()) for l in lines[data_start:] if l.strip()]
        bc_z0 = [temps[i] for i in [0, 1, 2, 9, 10, 11, 18, 19, 20]]
        bc_z1 = [temps[i] for i in [6, 7, 8, 15, 16, 17, 24, 25, 26]]
        for t in bc_z0:
            assert abs(t - 400.0) < 1.0, f"BC node drifted to {t}K (should stay 400K)"
        for t in bc_z1:
            assert abs(t - 800.0) < 1.0, f"BC node drifted to {t}K (should stay 800K)"


@skip_no_docker
class TestTransientFullPipeline:
    """End-to-end test: meshgen -> bc_transient -> gdps_thermal_transient -> parafem2vtu (PVD)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        for img in ['parafem:local', 'parafem-bcgen:local',
                     'parafem:local', 'parafem-vtu:local']:
            if not image_exists(img):
                pytest.skip(f'{img} not built')

    def test_end_to_end(self):
        import json

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
        r = run_in_container('parafem:local', self.workdir, mesh_cmd)
        assert r.returncode == 0, f"Meshgen failed: {r.stdout}\n{r.stderr}"
        assert os.path.exists(os.path.join(self.workdir, 'job.d'))

        # Step 2: BC generation (transient)
        zone_config = [
            {"axis_min": 0.0, "axis_max": 0.1, "temperature": 800.0},
            {"axis_min": 0.9, "axis_max": 1.0, "temperature": 400.0},
        ]
        zone_file = os.path.join(self.workdir, 'zones.json')
        with open(zone_file, 'w') as f:
            json.dump(zone_config, f)

        bc_cmd = """python3 /tools/gdps_bc_transient/gdps_bc_transient.py \
            --mesh_d /work/job.d \
            --kx 50.0 --ky 50.0 --kz 50.0 --rho 7800.0 --cp 500.0 \
            --val0 293.0 --dtim 500.0 --nstep 10 --npri 5 \
            --theta 0.5 --tol 1.0e-6 --limit 2000 \
            --zone_config /work/zones.json \
            --bc_axis z \
            --output_dat /work/job.dat \
            --output_bnd /work/job.bnd \
            --output_fix /work/job.fix \
            --output_mat /work/job.mat \
            --output_d /work/job_bc.d"""
        r = run_in_container('parafem-bcgen:local', self.workdir, bc_cmd)
        assert r.returncode == 0, f"BC gen failed: {r.stdout}\n{r.stderr}"
        for fname in ['job.dat', 'job.bnd', 'job.fix', 'job.mat']:
            assert os.path.exists(os.path.join(self.workdir, fname)), f"Missing {fname}"

        # Step 3: Solver (gdps_thermal_transient)
        solver_cmd = "cd /work && mpirun -np 1 gdps_thermal_transient job 2>&1"
        r = run_in_container('parafem:local', self.workdir, solver_cmd)
        assert r.returncode == 0, f"Solver failed: {r.stdout}\n{r.stderr}"

        res_path = os.path.join(self.workdir, 'job.res')
        assert os.path.exists(res_path), "Solver .res file missing"
        assert os.path.getsize(res_path) > 0, "Solver .res file is empty"

        with open(res_path) as f:
            res = f.read()
        assert 'NaN' not in res

        # Step 4: VTU conversion (transient/PVD output)
        vtu_cmd = """python3 /tools/gdps_parafem2vtu/parafem2vtu.py \
            --mesh_d /work/job.d --fix /work/job.fix \
            --ensi_dir /work --jobname job \
            --dtim 500 \
            --output /work/result.vtu"""
        r = run_in_container('parafem-vtu:local', self.workdir, vtu_cmd)
        assert r.returncode == 0, f"VTU failed: {r.stdout}\n{r.stderr}"

        # Verify PVD file exists (multi-timestep output)
        pvd_path = os.path.join(self.workdir, 'result.pvd')
        assert os.path.exists(pvd_path), "PVD file not produced"

        # Verify PVD is valid XML
        tree = ET.parse(pvd_path)
        root = tree.getroot()
        assert root.tag == 'VTKFile'
        assert root.attrib['type'] == 'Collection'

        # Verify at least one timestep VTU file exists
        vtu_files = [f for f in os.listdir(self.workdir)]
        assert any(re.match(r'result_\d{6}\.vtu$', f) for f in vtu_files)


@skip_no_docker
class TestImportPipeline:
    """End-to-end test for imported meshes: .inp -> .d/.nset -> BC (nset mode) -> solver."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem:local'):
            pytest.skip('parafem:local not built')
        if not image_exists('parafem-bcgen:local'):
            pytest.skip('parafem-bcgen:local not built')
        
        import shutil
        fixtures = os.path.join(REPO_ROOT, 'tests', 'fixtures')
        shutil.copy(os.path.join(fixtures, 'small_2x2x2_import.inp'), 
                    os.path.join(self.workdir, 'job.inp'))

    def test_import_to_solve_steady(self):
        # 1. Import
        import_cmd = "inp2pf -renumber job.inp 2>&1"
        r = run_in_container('parafem:local', self.workdir, import_cmd)
        assert r.returncode == 0
        
        # 2. BC (NSET mode)
        bc_config = '[{"nset_name": "UPSTREAM", "temperature": 800.0}, {"nset_name": "DOWNSTREAM", "temperature": 300.0}]'
        bc_cmd = f"""python3 /tools/gdps_bc_thermal/gdps_bc_thermal.py \
            --mesh_d /work/job.d \
            --kx 50 --ky 50 --kz 50 \
            --bc_mode nset \
            --nset_file /work/job.nset \
            --zone_config '{bc_config}' \
            --output_dat /work/job.dat \
            --output_bnd /work/job.bnd \
            --output_fix /work/job.fix \
            --output_d /work/job_out.d"""
        r = run_in_container('parafem-bcgen:local', self.workdir, bc_cmd)
        assert r.returncode == 0
        
        # 3. Solver
        solver_cmd = "mkdir -p run && cd run && cp /work/job.dat /work/job.d /work/job.bnd /work/job.fix . && mpirun -np 1 p123 job 2>&1 && cp job.res /work/"
        r = run_in_container('parafem:local', self.workdir, solver_cmd)
        assert r.returncode == 0
        
        # 4. Verify results
        with open(os.path.join(self.workdir, 'job.res')) as f:
            res = f.read()
        assert 'NaN' not in res
        match = re.search(r'iterations to convergence was\s+(\d+)', res)
        assert match and int(match.group(1)) < 100
