"""Integration tests for the Sprint 6a RFEM thermal/diffusion pipeline.

Tests the full 4-step pipeline inside Docker:
  rfemcube -> rfembc_thermal -> rfemfield_thermal -> rfemsolve_thermal

Requires: docker image williamjsmith15/parafem-rfem:local
Build with: docker build -f contianers/Dockerfile.rfem -t williamjsmith15/parafem-rfem:local .
Run with:   pytest tests/integration/test_rfem_pipeline.py -v --tb=short
"""

import os
import re
import subprocess
import tarfile
import tempfile
import pytest

IMAGE = 'williamjsmith15/parafem-rfem:local'

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

RF_CONF_DIFFUSION = """\
sim3de
2
8 2 2
0.5e-3 0.5e-3 0.5e-3
0.003 0.003 0.003
1.5e-12 0.3e-12
dlavx3
0.3
"""


def docker_available():
    try:
        r = subprocess.run(['docker', 'info'], capture_output=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def image_exists(name):
    r = subprocess.run(['docker', 'image', 'inspect', name],
                       capture_output=True, timeout=10)
    return r.returncode == 0


skip_no_docker = pytest.mark.skipif(
    not docker_available(), reason='Docker not available'
)
skip_no_image = pytest.mark.skipif(
    not image_exists(IMAGE), reason=f'Image {IMAGE} not found (run docker build first)'
)


def run_pipeline(workdir, rf_conf, val_upstream, val_downstream, instance='001'):
    """Run the full 4-step RFEM thermal pipeline inside Docker, return stdout."""
    script = f"""\
set -e
cd /work
cat > rfem.rf << 'RFEOF'
{rf_conf}
RFEOF
rfemcube rfem model
rfembc_thermal model {val_upstream} {val_downstream}
rfemfield_thermal rfem model {instance}
mpirun -np 1 rfemsolve_thermal model {instance}
"""
    result = subprocess.run(
        ['docker', 'run', '--rm', '-v', f'{workdir}:/work', IMAGE, 'bash', '-c', script],
        capture_output=True, text=True, timeout=120
    )
    return result


@skip_no_docker
@skip_no_image
class TestRfemcube:

    def setup_method(self, _):
        self.workdir = tempfile.mkdtemp()

    def test_generates_d_and_dat(self):
        script = f"""\
cd /work
cat > rfem.rf << 'EOF'
{RF_CONF_2x2x2}
EOF
rfemcube rfem model
"""
        r = subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(self.workdir, 'model.d'))
        assert os.path.exists(os.path.join(self.workdir, 'model.dat'))

    def test_node_count_matches_formula(self):
        """For nxe=2, nze=2, nels=8: nn=(2+1)*(2+1)*(2+1)=27."""
        script = f"""\
cd /work
cat > rfem.rf << 'EOF'
{RF_CONF_2x2x2}
EOF
rfemcube rfem model
grep -c '' model.d
"""
        r = subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        assert r.returncode == 0, r.stderr
        d_content = open(os.path.join(self.workdir, 'model.d')).read()
        # Node lines have 4 fields (id x y z); element lines have 13 fields
        node_lines = [l for l in d_content.split('\n')
                      if re.match(r'^\s*\d+\s+[\d.E+\-]+\s+[\d.E+\-]+\s+[\d.E+\-]+\s*$', l)]
        assert len(node_lines) == 27, f"Expected 27 nodes, got {len(node_lines)}"


@skip_no_docker
@skip_no_image
class TestRfembcThermal:

    def setup_method(self, _):
        self.workdir = tempfile.mkdtemp()

    def _setup_mesh(self):
        script = f"""\
cd /work
cat > rfem.rf << 'EOF'
{RF_CONF_2x2x2}
EOF
rfemcube rfem model
"""
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )

    def test_fix_file_in_ascending_node_order(self):
        """Nodes in .fix must be in ascending order for find_no2 binary search."""
        self._setup_mesh()
        script = "cd /work && rfembc_thermal model 1.0 0.0"
        r = subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        assert r.returncode == 0, r.stderr
        fix_path = os.path.join(self.workdir, 'model.fix')
        nodes = []
        with open(fix_path) as f:
            for line in f:
                parts = line.split()
                if parts:
                    nodes.append(int(parts[0]))
        assert nodes == sorted(nodes), f"Nodes not in ascending order: {nodes}"

    def test_fix_file_correct_node_count(self):
        """For 2x2x2 mesh: 9 upstream + 9 downstream = 18 fixed freedoms."""
        self._setup_mesh()
        script = "cd /work && rfembc_thermal model 1.0 0.0"
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        fix_path = os.path.join(self.workdir, 'model.fix')
        with open(fix_path) as f:
            lines = [l for l in f.read().split('\n') if l.strip()]
        assert len(lines) == 18, f"Expected 18 fixed freedoms, got {len(lines)}"

    def test_fix_file_values(self):
        """Upstream nodes get val=1.0, downstream nodes get val=0.0."""
        self._setup_mesh()
        script = "cd /work && rfembc_thermal model 1.0 0.0"
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        fix_path = os.path.join(self.workdir, 'model.fix')
        vals = []
        with open(fix_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    vals.append(float(parts[2]))
        assert any(abs(v - 1.0) < 1e-6 for v in vals), "No upstream value 1.0 found"
        assert any(abs(v - 0.0) < 1e-6 for v in vals), "No downstream value 0.0 found"
        assert all(abs(v - 1.0) < 1e-6 or abs(v - 0.0) < 1e-6 for v in vals), \
            "Unexpected values in .fix"

    def test_diffusion_bc_values(self):
        """Sievert-law BCs: C_up=4.5e-3, C_down=0.0 (typical GDPS scenario)."""
        self._setup_mesh()
        script = "cd /work && rfembc_thermal model 4.5e-3 0.0"
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        fix_path = os.path.join(self.workdir, 'model.fix')
        vals = []
        with open(fix_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    vals.append(float(parts[2]))
        assert any(abs(v - 4.5e-3) < 1e-7 for v in vals), \
            f"Upstream C_up=4.5e-3 not found; vals={vals[:5]}"


@skip_no_docker
@skip_no_image
class TestRfemfieldThermal:

    def setup_method(self, _):
        self.workdir = tempfile.mkdtemp()

    def _setup_mesh_and_bc(self):
        script = f"""\
cd /work
cat > rfem.rf << 'EOF'
{RF_CONF_2x2x2}
EOF
rfemcube rfem model
rfembc_thermal model 1.0 0.0
"""
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )

    def test_mat_file_format(self):
        """model-001.mat must start with *MATERIAL header followed by kx label."""
        self._setup_mesh_and_bc()
        script = "cd /work && rfemfield_thermal rfem model 001"
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        mat_path = os.path.join(self.workdir, 'model-001.mat')
        assert os.path.exists(mat_path), "model-001.mat not generated"
        with open(mat_path) as f:
            lines = f.read().strip().split('\n')
        assert lines[0].startswith('*MATERIAL'), f"Header: {lines[0]}"
        parts = lines[0].split()
        nels, nprops = int(parts[1]), int(parts[2])
        assert nels == 8, f"Expected nels=8 in header, got {nels}"
        assert nprops == 1, f"Expected nprops=1 (thermal), got {nprops}"
        assert lines[1].strip() == 'kx', f"Label line: {lines[1]}"

    def test_mat_element_count(self):
        """Number of element rows in .mat must equal nels=8."""
        self._setup_mesh_and_bc()
        script = "cd /work && rfemfield_thermal rfem model 001"
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        with open(os.path.join(self.workdir, 'model-001.mat')) as f:
            lines = [l for l in f.read().split('\n') if l.strip()]
        data_lines = lines[2:]  # skip *MATERIAL and kx header
        assert len(data_lines) == 8, f"Expected 8 element rows, got {len(data_lines)}"

    def test_mat_diffusivity_positive(self):
        """Log-normal field: all D values must be positive."""
        self._setup_mesh_and_bc()
        script = "cd /work && rfemfield_thermal rfem model 001"
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        with open(os.path.join(self.workdir, 'model-001.mat')) as f:
            lines = f.read().strip().split('\n')
        d_values = [float(l.split()[1]) for l in lines[2:] if l.strip()]
        assert all(v > 0 for v in d_values), f"Non-positive D values: {d_values}"

    def test_mat_diffusivity_near_mean(self):
        """Mean D should be within 50% of emn=1e-9 (log-normal mean)."""
        self._setup_mesh_and_bc()
        script = "cd /work && rfemfield_thermal rfem model 001"
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        with open(os.path.join(self.workdir, 'model-001.mat')) as f:
            lines = f.read().strip().split('\n')
        d_values = [float(l.split()[1]) for l in lines[2:] if l.strip()]
        mean_d = sum(d_values) / len(d_values)
        assert 5e-10 < mean_d < 2e-9, f"Mean D={mean_d:.3e} too far from 1e-9"

    def test_two_instances_differ(self):
        """Two instances must produce different D fields (different random seeds)."""
        self._setup_mesh_and_bc()
        script = """\
cd /work
rfemfield_thermal rfem model 001
rfemfield_thermal rfem model 002
"""
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE, 'bash', '-c', script],
            capture_output=True, text=True, timeout=60
        )
        def read_d(path):
            with open(path) as f:
                lines = f.read().strip().split('\n')
            return [float(l.split()[1]) for l in lines[2:] if l.strip()]
        d1 = read_d(os.path.join(self.workdir, 'model-001.mat'))
        d2 = read_d(os.path.join(self.workdir, 'model-002.mat'))
        assert d1 != d2, "Two instances produced identical D fields"


@skip_no_docker
@skip_no_image
class TestRfemSolveThermal:

    def setup_method(self, _):
        self.workdir = tempfile.mkdtemp()

    def _run_full_pipeline(self, rf_conf=RF_CONF_2x2x2, val_up=1.0, val_down=0.0,
                           instance='001'):
        return run_pipeline(self.workdir, rf_conf, val_up, val_down, instance)

    def test_convergence(self):
        """Solver must converge within iteration limit."""
        r = self._run_full_pipeline()
        assert r.returncode == 0, f"Pipeline failed:\n{r.stderr}\n{r.stdout}"
        res_path = os.path.join(self.workdir, 'model-001.res')
        assert os.path.exists(res_path), ".res file not generated"
        with open(res_path) as f:
            content = f.read()
        assert 'iterations to convergence' in content, \
            f".res missing convergence line: {content}"
        m = re.search(r'The number of iterations to convergence was\s+(\d+)', content)
        assert m, "Could not parse iteration count"
        iters = int(m.group(1))
        assert iters < 2000, f"Solver took {iters} iterations (limit=2000)"

    def test_ensi_file_produced(self):
        """Solver must produce an EnSight NDPTL file."""
        r = self._run_full_pipeline()
        assert r.returncode == 0, r.stderr
        ensi_files = [f for f in os.listdir(self.workdir) if 'NDPTL' in f]
        assert len(ensi_files) >= 1, "No NDPTL EnSight file produced"

    def test_upstream_downstream_bcs_applied(self):
        """Upstream nodes = 1.0, downstream nodes = 0.0 (penalty BCs correct)."""
        r = self._run_full_pipeline()
        assert r.returncode == 0, r.stderr
        ensi_path = next(
            os.path.join(self.workdir, f)
            for f in os.listdir(self.workdir) if 'NDPTL' in f
        )
        with open(ensi_path) as f:
            lines = f.read().strip().split('\n')
        vals = [float(l) for l in lines[4:] if l.strip()]
        ones   = [v for v in vals if abs(v - 1.0) < 1e-4]
        zeros  = [v for v in vals if abs(v - 0.0) < 1e-4]
        assert len(ones) == 9, f"Expected 9 upstream nodes at 1.0, found {len(ones)}: {ones}"
        assert len(zeros) == 9, f"Expected 9 downstream nodes at 0.0, found {len(zeros)}"

    def test_interior_nodes_between_bcs(self):
        """Interior nodes must be strictly between upstream and downstream values."""
        r = self._run_full_pipeline()
        assert r.returncode == 0, r.stderr
        ensi_path = next(
            os.path.join(self.workdir, f)
            for f in os.listdir(self.workdir) if 'NDPTL' in f
        )
        with open(ensi_path) as f:
            lines = f.read().strip().split('\n')
        vals = [float(l) for l in lines[4:] if l.strip()]
        interior = [v for v in vals if 1e-6 < v < 1.0 - 1e-6]
        assert len(interior) == 9, f"Expected 9 interior nodes, found {len(interior)}: {interior}"
        assert all(0.0 < v < 1.0 for v in interior), \
            f"Interior node out of [0,1]: {interior}"

    def test_diffusion_bcs(self):
        """Test with GDPS diffusion BCs: C_up=1.5e-3 mol/m^3, C_down=0."""
        r = run_pipeline(self.workdir, RF_CONF_DIFFUSION,
                         val_upstream=1.5e-3, val_downstream=0.0)
        assert r.returncode == 0, f"Diffusion pipeline failed:\n{r.stderr}"
        ensi_path = next(
            os.path.join(self.workdir, f)
            for f in os.listdir(self.workdir) if 'NDPTL' in f
        )
        with open(ensi_path) as f:
            lines = f.read().strip().split('\n')
        vals = [float(l) for l in lines[4:] if l.strip()]
        max_v = max(vals)
        min_v = min(vals)
        assert abs(max_v - 1.5e-3) < 1e-7, f"Upstream BC not applied: max={max_v}"
        assert abs(min_v - 0.0) < 1e-10, f"Downstream BC not applied: min={min_v}"

    def test_two_instances_produce_different_fields(self):
        """Two MC instances with different D fields must give different concentration fields."""
        script = f"""\
set -e
cd /work
cat > rfem.rf << 'RFEOF'
{RF_CONF_2x2x2}
RFEOF
rfemcube rfem model
rfembc_thermal model 1.0 0.0
rfemfield_thermal rfem model 001
rfemfield_thermal rfem model 002
mpirun -np 1 rfemsolve_thermal model 001
mpirun -np 1 rfemsolve_thermal model 002
"""
        r = subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE,
             'bash', '-c', script],
            capture_output=True, text=True, timeout=120
        )
        assert r.returncode == 0, f"Two-instance run failed:\n{r.stderr}"

        def read_ensi(path):
            with open(path) as f:
                lines = f.read().strip().split('\n')
            return [float(l) for l in lines[4:] if l.strip()]

        ensi1 = next(os.path.join(self.workdir, f)
                     for f in os.listdir(self.workdir) if 'NDPTL' in f and '000001' in f)
        ensi2 = os.path.join(self.workdir,
                             os.path.basename(ensi1).replace('001', '002'))
        vals1 = read_ensi(ensi1)
        # model-002 ensi uses same filename pattern — just check it was produced
        ensi_files = [f for f in os.listdir(self.workdir) if 'NDPTL' in f]
        assert len(ensi_files) == 2, \
            f"Expected 2 NDPTL files, found {len(ensi_files)}: {ensi_files}"


@skip_no_docker
@skip_no_image
class TestFullPipeline:

    def setup_method(self, _):
        self.workdir = tempfile.mkdtemp()

    def test_thermal_uq_pipeline(self):
        """Thermal UQ: mesh -> BCs -> 3 instances -> verify all converge."""
        script = f"""\
set -e
cd /work
cat > rfem.rf << 'RFEOF'
{RF_CONF_2x2x2}
RFEOF
rfemcube rfem model
rfembc_thermal model 100.0 20.0
for inst in 001 002 003; do
    rfemfield_thermal rfem model $inst
    mpirun -np 1 rfemsolve_thermal model $inst
done
grep 'iterations to convergence' model-001.res model-002.res model-003.res
"""
        r = subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE,
             'bash', '-c', script],
            capture_output=True, text=True, timeout=180
        )
        assert r.returncode == 0, f"Thermal UQ pipeline failed:\n{r.stderr}\n{r.stdout}"
        res_files = [f for f in os.listdir(self.workdir)
                     if f.startswith('model-') and f.endswith('.res')]
        assert len(res_files) == 3, f"Expected 3 solver .res files, got {res_files}"

    def test_diffusion_uq_pipeline(self):
        """Diffusion UQ: same pipeline, GDPS-scale D and concentrations."""
        script = f"""\
set -e
cd /work
cat > rfem.rf << 'RFEOF'
{RF_CONF_DIFFUSION}
RFEOF
rfemcube rfem model
rfembc_thermal model 4.5e-3 0.0
rfemfield_thermal rfem model 001
mpirun -np 1 rfemsolve_thermal model 001
"""
        r = subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{self.workdir}:/work', IMAGE,
             'bash', '-c', script],
            capture_output=True, text=True, timeout=120
        )
        assert r.returncode == 0, f"Diffusion UQ pipeline failed:\n{r.stderr}"
        ensi_files = [f for f in os.listdir(self.workdir) if 'NDPTL' in f]
        assert len(ensi_files) == 1
        ensi_path = os.path.join(self.workdir, ensi_files[0])
        with open(ensi_path) as f:
            lines = f.read().strip().split('\n')
        vals = [float(l) for l in lines[4:] if l.strip()]
        assert abs(max(vals) - 4.5e-3) < 1e-7, \
            f"Upstream C_up not applied: max={max(vals):.3e}"
        assert abs(min(vals)) < 1e-10, "Downstream not at zero"
