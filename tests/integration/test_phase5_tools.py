"""Integration tests for Phase 5: gdps_sievert_bc and gdps_compute_diffusivity tools.

Requires parafem-bcgen:local Docker image.
Run with: pytest tests/integration/test_phase5_tools.py -v
"""

import math
import os
import shutil
import subprocess
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TOOLS_DIR = os.path.join(REPO_ROOT, 'galaxy', 'tools')
FIXTURES_DIR = os.path.join(REPO_ROOT, 'tests', 'fixtures')

R_GAS = 8.314462618
# Iron parameters from materials.json
IRON_S0 = 5.1e-3
IRON_ES = 28600.0
IRON_D0 = 4.1e-8
IRON_EA = 4100.0


def docker_available():
    try:
        result = subprocess.run(['docker', 'info'], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def image_exists(name):
    result = subprocess.run(['docker', 'image', 'inspect', name], capture_output=True, timeout=10)
    return result.returncode == 0


skip_no_docker = pytest.mark.skipif(
    not docker_available(), reason='Docker not available'
)


def run_in_container(image, workdir, command, timeout=60):
    return subprocess.run(
        ['docker', 'run', '--rm',
         '-v', f'{workdir}:/work',
         '-v', f'{TOOLS_DIR}:/tools',
         image, 'bash', '-c', command],
        capture_output=True, text=True, timeout=timeout
    )


@skip_no_docker
class TestSievertBC:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem-bcgen:local'):
            pytest.skip('parafem-bcgen:local not built')
        shutil.copy(os.path.join(FIXTURES_DIR, 'small_2x2x2.d'),
                    os.path.join(self.workdir, 'mesh.d'))
        shutil.copy(os.path.join(FIXTURES_DIR, 'small_2x2x2_import.nset'),
                    os.path.join(self.workdir, 'mesh.nset'))

    def _run_static(self, upstream_p=100000.0, downstream_p=0.0, temp=500.0):
        cmd = (
            f"python3 /tools/gdps_sievert_bc/gdps_sievert_bc.py "
            f"--mesh_d /work/mesh.d "
            f"--bc_mode nset "
            f"--nset_file /work/mesh.nset "
            f"--upstream_nset UPSTREAM "
            f"--downstream_nset DOWNSTREAM "
            f"--S0 {IRON_S0} --Es {IRON_ES} "
            f"--temp_mode scalar --temp_scalar {temp} "
            f"--upstream_p {upstream_p} --downstream_p {downstream_p} "
            f"--output_fix /work/out.fix"
        )
        return run_in_container('parafem-bcgen:local', self.workdir, cmd)

    def test_static_fix_created(self):
        result = self._run_static()
        assert result.returncode == 0, f"sievert_bc failed:\n{result.stdout}\n{result.stderr}"
        assert os.path.exists(os.path.join(self.workdir, 'out.fix'))

    def test_static_fix_node_count(self):
        self._run_static()
        with open(os.path.join(self.workdir, 'out.fix')) as f:
            lines = [l for l in f if l.strip()]
        # 9 upstream + 9 downstream = 18 fixed nodes
        assert len(lines) == 18, f"Expected 18 fixed nodes, got {len(lines)}"

    def test_static_fix_upstream_concentration(self):
        T = 500.0
        P = 100000.0
        self._run_static(upstream_p=P, downstream_p=0.0, temp=T)
        S = IRON_S0 * math.exp(-IRON_ES / (R_GAS * T))
        expected_c_up = S * math.sqrt(P)

        with open(os.path.join(self.workdir, 'out.fix')) as f:
            lines = [l.strip() for l in f if l.strip()]

        # First 9 lines are upstream nodes (sorted by node_id, nodes 1-9)
        for line in lines[:9]:
            parts = line.split()
            val = float(parts[2])
            assert abs(val - expected_c_up) / expected_c_up < 1e-6, \
                f"Upstream concentration {val} != expected {expected_c_up}"

    def test_static_fix_downstream_zero(self):
        self._run_static(downstream_p=0.0)
        with open(os.path.join(self.workdir, 'out.fix')) as f:
            lines = [l.strip() for l in f if l.strip()]
        # Last 9 lines are downstream nodes (nodes 19-27), downstream_p=0 → concentration=0
        for line in lines[9:]:
            parts = line.split()
            val = float(parts[2])
            assert val == 0.0, f"Downstream concentration should be 0, got {val}"

    def test_scheduled_produces_fix_and_bcs(self):
        schedule = '[{"step": 1, "upstream_p": 100000.0, "downstream_p": 0.0}, {"step": 5, "upstream_p": 200000.0, "downstream_p": 0.0}]'
        cmd = (
            f"python3 /tools/gdps_sievert_bc/gdps_sievert_bc.py "
            f"--mesh_d /work/mesh.d "
            f"--bc_mode nset "
            f"--nset_file /work/mesh.nset "
            f"--upstream_nset UPSTREAM "
            f"--downstream_nset DOWNSTREAM "
            f"--S0 {IRON_S0} --Es {IRON_ES} "
            f"--temp_mode scalar --temp_scalar 500.0 "
            f"--pressure_schedule '{schedule}' "
            f"--output_fix /work/out.fix "
            f"--output_bcs /work/out.bcs"
        )
        result = run_in_container('parafem-bcgen:local', self.workdir, cmd)
        assert result.returncode == 0, f"sievert_bc scheduled failed:\n{result.stdout}\n{result.stderr}"
        assert os.path.exists(os.path.join(self.workdir, 'out.fix'))
        assert os.path.exists(os.path.join(self.workdir, 'out.bcs'))

    def test_scheduled_bcs_has_two_blocks(self):
        self.test_scheduled_produces_fix_and_bcs()
        with open(os.path.join(self.workdir, 'out.bcs')) as f:
            content = f.read()
        # Two schedule entries → two step-header blocks
        step_headers = [l.strip() for l in content.splitlines() if l.strip().isdigit()]
        assert len(step_headers) == 2, f"Expected 2 step blocks, got {len(step_headers)}"

    def test_zone_mode_finds_boundary_nodes(self):
        # The 2x2x2 fixture z-axis goes 0 to -1.0; z=0 face = nodes 1,2,3,10,11,12,19,20,21
        # zone upstream_min=0.95, upstream_max=1.0 captures these (normalised from -1 to 0)
        cmd = (
            f"python3 /tools/gdps_sievert_bc/gdps_sievert_bc.py "
            f"--mesh_d /work/mesh.d "
            f"--bc_mode zone "
            f"--bc_axis z "
            f"--upstream_min 0.0 --upstream_max 0.05 "
            f"--downstream_min 0.95 --downstream_max 1.0 "
            f"--S0 {IRON_S0} --Es {IRON_ES} "
            f"--temp_mode scalar --temp_scalar 500.0 "
            f"--upstream_p 100000.0 --downstream_p 0.0 "
            f"--output_fix /work/out_zone.fix"
        )
        result = run_in_container('parafem-bcgen:local', self.workdir, cmd)
        assert result.returncode == 0, f"zone mode failed:\n{result.stdout}\n{result.stderr}"
        with open(os.path.join(self.workdir, 'out_zone.fix')) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 18, f"Expected 18 fixed nodes (9 upstream + 9 downstream), got {len(lines)}"


@skip_no_docker
class TestComputeDiffusivity:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem-bcgen:local'):
            pytest.skip('parafem-bcgen:local not built')
        shutil.copy(os.path.join(FIXTURES_DIR, 'small_2x2x2.d'),
                    os.path.join(self.workdir, 'mesh.d'))
        shutil.copy(os.path.join(FIXTURES_DIR, 'small_2x2x2.dat'),
                    os.path.join(self.workdir, 'mesh.dat'))

    def _run(self, temp=500.0, nstep=1, dtim=1e10, theta=1.0, val0=0.0):
        cmd = (
            f"python3 /tools/gdps_compute_diffusivity/gdps_compute_diffusivity.py "
            f"--mesh_d /work/mesh.d "
            f"--dat_file /work/mesh.dat "
            f"--D0 {IRON_D0} --Ea {IRON_EA} "
            f"--temp_mode scalar --temp_scalar {temp} "
            f"--nstep {nstep} --dtim {dtim} --theta {theta} --val0 {val0} "
            f"--output_mat /work/out.mat "
            f"--output_d /work/out.d "
            f"--output_dat /work/out.dat"
        )
        return run_in_container('parafem-bcgen:local', self.workdir, cmd)

    def test_outputs_created(self):
        result = self._run()
        assert result.returncode == 0, f"compute_diffusivity failed:\n{result.stdout}\n{result.stderr}"
        assert os.path.exists(os.path.join(self.workdir, 'out.mat'))
        assert os.path.exists(os.path.join(self.workdir, 'out.d'))
        assert os.path.exists(os.path.join(self.workdir, 'out.dat'))

    def test_mat_has_nels_entries(self):
        self._run()
        with open(os.path.join(self.workdir, 'out.mat')) as f:
            lines = [l.strip() for l in f if l.strip()]
        # Header: *MATERIAL 8 5 + comment + 8 data lines = 10 total lines
        header_parts = lines[0].split()
        nels_in_header = int(header_parts[1])
        assert nels_in_header == 8, f"Expected 8 material entries, got {nels_in_header}"
        data_lines = [l for l in lines[2:] if not l.startswith('*') and not l.startswith('ID')]
        assert len(data_lines) == 8, f"Expected 8 data lines, got {len(data_lines)}"

    def test_mat_diffusivity_value(self):
        T = 500.0
        self._run(temp=T)
        expected_D = IRON_D0 * math.exp(-IRON_EA / (R_GAS * T))
        with open(os.path.join(self.workdir, 'out.mat')) as f:
            lines = [l.strip() for l in f if l.strip()]
        # First data line after header+comment: id kx ky kz rho cp
        data_line = lines[2]
        parts = data_line.split()
        kx = float(parts[1])
        assert abs(kx - expected_D) / expected_D < 1e-6, \
            f"kx={kx} != expected D={expected_D}"

    def test_d_etype_pp_updated(self):
        self._run()
        with open(os.path.join(self.workdir, 'out.d')) as f:
            content = f.read()
        in_elements = False
        elem_id = 0
        for line in content.splitlines():
            if '*ELEMENTS' in line:
                in_elements = True
                continue
            if not in_elements or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            elem_id += 1
            etype_pp = int(parts[3])
            assert etype_pp == elem_id, \
                f"Element {elem_id}: expected etype_pp={elem_id}, got {etype_pp}"

    def test_dat_is_p124_format(self):
        self._run(nstep=5, dtim=10.0, theta=0.5, val0=300.0)
        with open(os.path.join(self.workdir, 'out.dat')) as f:
            lines = [l.strip() for l in f if l.strip()]
        # Line 0: element type
        assert lines[0].startswith("'"), f"Line 0 should be element type: {lines[0]}"
        # Line 3: integer params — should have 8 tokens (np_types nels nn nr nip nod loaded fixed)
        int_parts = lines[3].split()
        assert len(int_parts) == 8, f"Integer params line should have 8 tokens, got {len(int_parts)}: {lines[3]}"
        np_types = int(int_parts[0])
        nels = int(int_parts[1])
        assert np_types == nels, f"np_types ({np_types}) should equal nels ({nels})"

    def test_dat_p124_float_line(self):
        nstep = 7
        dtim = 0.5
        theta = 0.75
        val0 = 100.0
        self._run(nstep=nstep, dtim=dtim, theta=theta, val0=val0)
        with open(os.path.join(self.workdir, 'out.dat')) as f:
            lines = [l.strip() for l in f if l.strip()]
        # Line 4: float params — val0 dtim nstep npri theta tol limit nres
        float_parts = lines[4].split()
        assert len(float_parts) == 8, f"Float params line should have 8 tokens, got {len(float_parts)}"
        assert float(float_parts[0]) == pytest.approx(val0, rel=1e-4)
        assert float(float_parts[1]) == pytest.approx(dtim, rel=1e-4)
        assert int(float_parts[2]) == nstep
        assert float(float_parts[4]) == pytest.approx(theta, rel=1e-4)

    def test_dat_np_types_equals_nels(self):
        self._run()
        with open(os.path.join(self.workdir, 'out.dat')) as f:
            lines = [l.strip() for l in f if l.strip()]
        int_parts = lines[3].split()
        np_types = int(int_parts[0])
        nels = int(int_parts[1])
        assert np_types == 8
        assert nels == 8
        assert np_types == nels
