import os
import subprocess
import pytest
from tests.integration.test_pipeline import run_in_container, skip_no_docker, image_exists
from galaxy.tools.gdps_extract_ic.gdps_extract_ic import extract_ic

@skip_no_docker
class TestGDPSTransientIC:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.workdir = str(tmp_path)
        if not image_exists('parafem:local'):
            pytest.skip('parafem:local not built')

    def create_small_mesh(self):
        # Generate 2x2x2 mesh (nels=8, nxe=2, nze=2, nye=2)
        cmd = """
cat > job.mg << MGEOF
'p124'
'parafem' 8 2 2 8
1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
0.1 2 1.0
1 1.0e-6 100 10.0
1 0 0
MGEOF
p12meshgen job > /dev/null && chmod 777 /work/*
"""
        run_in_container('parafem:local', self.workdir, cmd)
        
        # Create .mat file (p12meshgen for p124 might not create it, or might create it differently)
        with open(os.path.join(self.workdir, 'job.mat'), 'w') as f:
            # *MATERIAL nmats nvals
            f.write("*MATERIAL 1 5\n")
            f.write("kx ky kz rho cp\n")
            f.write("1 1.0 1.0 1.0 1.0 1.0\n")

    def test_ic_from_file(self):
        self.create_small_mesh()
        
        # Run baseline: 2 steps, val0=10.0 (values are in the .mg file used in create_small_mesh)
        cmd = "mpirun -np 1 gdps_thermal_transient job && chmod 777 /work/*"
        result = run_in_container('parafem:local', self.workdir, cmd)
        assert result.returncode == 0
        
        # 2. Extract IC from step 2
        ensi_step2 = os.path.join(self.workdir, 'job.ensi.NDTTR-000002')
        assert os.path.exists(ensi_step2)
        ini_file = os.path.join(self.workdir, 'job_step2.ini')
        extract_ic(ensi_step2, ini_file)
        
        # 3. Run again with IC from file
        os.rename(os.path.join(self.workdir, 'job.res'), os.path.join(self.workdir, 'job_base.res'))
        
        with open(os.path.join(self.workdir, 'job.ctrl'), 'w') as f:
            f.write("2\n")
            f.write("job_step2.ini\n")
            
        # We can reuse the same .dat, but maybe we want only 1 step
        # To change only nstep, we could edit the file, or just use it as is.
        # Let's just use it as is, it will run 2 more steps.
            
        result = run_in_container('parafem:local', self.workdir, cmd)
        assert result.returncode == 0
        
        ensi_restart_step0 = os.path.join(self.workdir, 'job.ensi.NDTTR-000000')
        assert os.path.exists(ensi_restart_step0)
        
        with open(ensi_step2) as f1, open(ensi_restart_step0) as f2:
            assert f1.readlines() == f2.readlines()

    def test_restart_from_chk(self):
        self.create_small_mesh()
        
        cmd = "mpirun -np 1 gdps_thermal_transient job && chmod 777 /work/*"
        run_in_container('parafem:local', self.workdir, cmd)
        
        assert os.path.exists(os.path.join(self.workdir, 'job.chk'))
        ensi_step2 = os.path.join(self.workdir, 'job.ensi.NDTTR-000002')
        
        with open(os.path.join(self.workdir, 'job.ctrl'), 'w') as f:
            f.write("3\n")
            f.write("job.chk\n")
            
        # Run again. It will restart from step 2 and run up to step 2 (so 0 more steps?)
        # Wait, if j_chk = 2 and nstep = 2, the loop DO j=j_chk+1, nstep won't execute.
        # Let's increase nstep in the .dat file for the second run.
        dat_path = os.path.join(self.workdir, 'job.dat')
        with open(dat_path, 'r') as f:
            lines = f.readlines()
        
        # In p124 .dat:
        # line 1: element
        # line 2: mesh
        # line 3: partition
        # line 4: np_types
        # line 5: nels, nn, nr, nip, nod, loaded_nodes, fixed_freedoms
        # line 6: val0
        # line 7: dtim, nstep, npri, theta
        # ...
        parts = lines[6].split()
        parts[1] = "4" # increase nstep to 4
        lines[6] = " ".join(parts) + "\n"
        
        with open(dat_path, 'w') as f:
            f.writelines(lines)
            
        result = run_in_container('parafem:local', self.workdir, cmd)
        assert result.returncode == 0
        
        ensi_restart_step2 = os.path.join(self.workdir, 'job.ensi.NDTTR-000002')
        assert os.path.exists(ensi_restart_step2)
        
        with open(ensi_step2) as f1, open(ensi_restart_step2) as f2:
            assert f1.readlines() == f2.readlines()
        
        assert os.path.exists(os.path.join(self.workdir, 'job.ensi.NDTTR-000004'))

    def test_backward_compatibility(self):
        self.create_small_mesh()
        
        if os.path.exists(os.path.join(self.workdir, 'job.ctrl')):
            os.remove(os.path.join(self.workdir, 'job.ctrl'))
            
        cmd = "mpirun -np 1 gdps_thermal_transient job && chmod 777 /work/*"
        result = run_in_container('parafem:local', self.workdir, cmd)
        assert result.returncode == 0
        
        ensi_step0 = os.path.join(self.workdir, 'job.ensi.NDTTR-000000')
        with open(ensi_step0) as f:
            lines = f.readlines()
            assert float(lines[4].strip()) == pytest.approx(10.0) # val0 from .mg
