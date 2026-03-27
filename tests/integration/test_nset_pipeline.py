"""Integration tests for the NSET-based mesh import and BC assignment pipeline.

Requires Docker images to be built locally with :local tags:
  parafem:local
  parafem-bcgen:local

Run with: pytest tests/integration/test_nset_pipeline.py -v --tb=short
"""

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TOOLS_DIR = os.path.join(REPO_ROOT, 'galaxy', 'tools')
FIXTURES_DIR = os.path.join(REPO_ROOT, 'tests', 'fixtures')

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
class TestNsetPipeline:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem:local'):
            pytest.skip('parafem:local not built')
        if not image_exists('parafem-bcgen:local'):
            pytest.skip('parafem-bcgen:local not built')
        
        # Copy the .inp fixture with NSETs
        shutil.copy(os.path.join(FIXTURES_DIR, 'small_2x2x2_import.inp'), 
                    os.path.join(self.workdir, 'job.inp'))

    def test_nset_import_and_bc_assignment(self):
        # Step 1: Import .inp and generate .d and .nset
        import_cmd = "inp2pf -renumber job.inp 2>&1"
        r = run_in_container('parafem:local', self.workdir, import_cmd)
        assert r.returncode == 0, f"Import failed: {r.stdout}\n{r.stderr}"
        
        nset_path = os.path.join(self.workdir, 'job.nset')
        assert os.path.exists(nset_path)
        with open(nset_path) as f:
            nset_content = f.read()
        assert 'UPSTREAM' in nset_content
        assert 'DOWNSTREAM' in nset_content
        assert '*NSET 9 UPSTREAM' in nset_content

        # Step 2: Run BC generator in nset mode
        # Configuration for NSET-based BCs
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
        assert r.returncode == 0, f"BC gen failed: {r.stdout}\n{r.stderr}"
        
        # Step 3: Verify .fix file has 18 entries (9 nodes for UPSTREAM, 9 for DOWNSTREAM)
        fix_path = os.path.join(self.workdir, 'job.fix')
        assert os.path.exists(fix_path)
        with open(fix_path) as f:
            fix_lines = [l for l in f.readlines() if l.strip()]
        assert len(fix_lines) == 18
        
        # Verify temperatures
        temps = [float(l.split()[2]) for l in fix_lines]
        assert 800.0 in temps
        assert 300.0 in temps
        assert temps.count(800.0) == 9
        assert temps.count(300.0) == 9

    def test_nset_transient_pipeline(self):
        # Step 1: Import (re-use logic or just assume it works from first test)
        import_cmd = "inp2pf -renumber job.inp 2>&1"
        run_in_container('parafem:local', self.workdir, import_cmd)

        # Step 2: Run transient BC generator in nset mode
        bc_config = '[{"nset_name": "UPSTREAM", "temperature": 500.0}, {"nset_name": "DOWNSTREAM", "temperature": 200.0}]'
        bc_cmd = f"""python3 /tools/gdps_bc_transient/gdps_bc_transient.py \
            --mesh_d /work/job.d \
            --kx 50 --ky 50 --kz 50 --rho 7800 --cp 500 \
            --val0 293 --dtim 10 --nstep 100 --npri 10 \
            --bc_mode nset \
            --nset_file /work/job.nset \
            --zone_config '{bc_config}' \
            --output_dat /work/job.dat \
            --output_bnd /work/job.bnd \
            --output_fix /work/job.fix \
            --output_mat /work/job.mat \
            --output_d /work/job_out.d"""
        
        r = run_in_container('parafem-bcgen:local', self.workdir, bc_cmd)
        assert r.returncode == 0, f"Transient BC gen failed: {r.stdout}\n{r.stderr}"
        
        fix_path = os.path.join(self.workdir, 'job.fix')
        with open(fix_path) as f:
            fix_lines = [l for l in f.readlines() if l.strip()]
        assert len(fix_lines) == 18
        temps = [float(l.split()[2]) for l in fix_lines]
        assert 500.0 in temps
        assert 200.0 in temps

    def test_mesh_preview_vtu_generation(self):
        # Step 1: Import
        import_cmd = "inp2pf -renumber job.inp 2>&1"
        run_in_container('parafem:local', self.workdir, import_cmd)
        
        # Step 2: Run mesh preview
        preview_cmd = f"""python3 /tools/gdps_mesh_preview/gdps_mesh_preview.py \
            --mesh_d /work/job.d \
            --nset_file /work/job.nset \
            --output /work/preview.vtu"""
        
        r = run_in_container('parafem-bcgen:local', self.workdir, preview_cmd)
        assert r.returncode == 0, f"Preview failed: {r.stdout}\n{r.stderr}"
        
        # Step 3: Verify VTU output
        vtu_path = os.path.join(self.workdir, 'preview.vtu')
        assert os.path.exists(vtu_path)
        with open(vtu_path) as f:
            vtu_content = f.read()
        
        assert '<DataArray type="Float64" Name="SurfaceID" format="ascii">' in vtu_content
        # Small mesh has 27 nodes. Check we have 27 SurfaceID values.
        tree = ET.fromstring(vtu_content)
        surface_id_arr = tree.find('.//PointData/DataArray[@Name="SurfaceID"]')
        assert surface_id_arr is not None
        vals = [float(x) for x in surface_id_arr.text.strip().split()]
        assert len(vals) == 27
        # At least some should be non-zero (1.0 or 2.0)
        assert any(v > 0 for v in vals)
