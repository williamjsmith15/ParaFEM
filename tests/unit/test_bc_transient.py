"""Unit tests for the transient thermal BC generator (gdps_bc_transient.py)."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'galaxy', 'tools', 'gdps_bc_transient'))

from gdps_bc_transient import (
    parse_d_file,
    find_boundary_nodes,
    assign_zones,
    write_fix,
    write_dat,
    write_mat,
)


class TestWriteDat:

    def test_dat_element_type(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            path = f.name
        try:
            write_dat(path, nels=8, nn=27, nr=0, nod=8, fixed_freedoms=18,
                      val0=400.0, dtim=10.0, nstep=5, npri=1,
                      theta=0.5, tol=1e-8, limit=200, element_type='hexahedron')
            with open(path) as f:
                lines = f.readlines()
            assert "'hexahedron'" in lines[0]
        finally:
            os.unlink(path)

    def test_dat_mesh_partition_lines(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            path = f.name
        try:
            write_dat(path, nels=8, nn=27, nr=0, nod=8, fixed_freedoms=18,
                      val0=400.0, dtim=10.0, nstep=5, npri=1,
                      theta=0.5, tol=1e-8, limit=200, element_type='hexahedron')
            with open(path) as f:
                lines = f.readlines()
            assert lines[1].strip() == '2'
            assert lines[2].strip() == '1'
        finally:
            os.unlink(path)

    def test_dat_mesh_params_line(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            path = f.name
        try:
            write_dat(path, nels=8, nn=27, nr=0, nod=8, fixed_freedoms=18,
                      val0=400.0, dtim=10.0, nstep=5, npri=1,
                      theta=0.5, tol=1e-8, limit=200, element_type='hexahedron')
            with open(path) as f:
                lines = f.readlines()
            parts = lines[3].split()
            assert int(parts[0]) == 1    # np_types
            assert int(parts[1]) == 8    # nels
            assert int(parts[2]) == 27   # nn
            assert int(parts[3]) == 0    # nr
            assert int(parts[4]) == 8    # nip
            assert int(parts[5]) == 8    # nod
            assert int(parts[6]) == 0    # loaded_freedoms
            assert int(parts[7]) == 18   # fixed_freedoms
        finally:
            os.unlink(path)

    def test_dat_time_params_line(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            path = f.name
        try:
            write_dat(path, nels=8, nn=27, nr=0, nod=8, fixed_freedoms=18,
                      val0=400.0, dtim=10.0, nstep=5, npri=1,
                      theta=0.5, tol=1e-8, limit=200, element_type='hexahedron')
            with open(path) as f:
                lines = f.readlines()
            parts = lines[4].split()
            assert float(parts[0]) == pytest.approx(400.0)   # val0
            assert float(parts[1]) == pytest.approx(10.0)    # dtim
            assert int(parts[2]) == 5                         # nstep
            assert int(parts[3]) == 1                         # npri
            assert float(parts[4]) == pytest.approx(0.5)     # theta
            assert float(parts[5]) == pytest.approx(1e-8)    # tol
            assert int(parts[6]) == 200                       # limit
            assert int(parts[7]) == 1                         # nres
        finally:
            os.unlink(path)

    def test_dat_nip_hex8(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            path = f.name
        try:
            write_dat(path, nels=8, nn=27, nr=0, nod=8, fixed_freedoms=0,
                      val0=293.0, dtim=1.0, nstep=1, npri=1,
                      theta=1.0, tol=1e-6, limit=100, element_type='hexahedron')
            with open(path) as f:
                lines = f.readlines()
            parts = lines[3].split()
            assert int(parts[4]) == 8  # nip=8 for 8-node hex
        finally:
            os.unlink(path)


class TestWriteMat:

    def test_mat_header(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mat', delete=False) as f:
            path = f.name
        try:
            write_mat(path, kx=50.0, ky=50.0, kz=50.0, rho=7800.0, cp=500.0)
            with open(path) as f:
                lines = f.readlines()
            assert lines[0].startswith('*MATERIAL')
            assert '1' in lines[0]
            assert '5' in lines[0]
        finally:
            os.unlink(path)

    def test_mat_column_header(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mat', delete=False) as f:
            path = f.name
        try:
            write_mat(path, kx=50.0, ky=50.0, kz=50.0, rho=7800.0, cp=500.0)
            with open(path) as f:
                lines = f.readlines()
            assert 'kx' in lines[1]
            assert 'ky' in lines[1]
            assert 'kz' in lines[1]
            assert 'rho' in lines[1]
            assert 'cp' in lines[1]
        finally:
            os.unlink(path)

    def test_mat_values(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mat', delete=False) as f:
            path = f.name
        try:
            write_mat(path, kx=50.0, ky=30.0, kz=20.0, rho=7800.0, cp=500.0)
            with open(path) as f:
                lines = f.readlines()
            parts = lines[2].split()
            assert int(parts[0]) == 1           # material ID
            assert float(parts[1]) == pytest.approx(50.0)
            assert float(parts[2]) == pytest.approx(30.0)
            assert float(parts[3]) == pytest.approx(20.0)
            assert float(parts[4]) == pytest.approx(7800.0)
            assert float(parts[5]) == pytest.approx(500.0)
        finally:
            os.unlink(path)

    def test_mat_line_count(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mat', delete=False) as f:
            path = f.name
        try:
            write_mat(path, kx=1.0, ky=1.0, kz=1.0, rho=1.0, cp=1.0)
            with open(path) as f:
                lines = [l for l in f.readlines() if l.strip()]
            # header + column names + 1 material row
            assert len(lines) == 3
        finally:
            os.unlink(path)


class TestWriteFix:

    def test_fix_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fix', delete=False) as f:
            path = f.name
        try:
            write_fix(path, {1: 800.0, 5: 400.0, 10: 600.0})
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3
            parts0 = lines[0].split()
            assert int(parts0[0]) == 1
            assert int(parts0[1]) == 1   # sense=1
            assert float(parts0[2]) == pytest.approx(800.0)
        finally:
            os.unlink(path)

    def test_fix_sorted_by_node(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fix', delete=False) as f:
            path = f.name
        try:
            write_fix(path, {10: 400.0, 2: 800.0, 7: 500.0})
            with open(path) as f:
                lines = f.readlines()
            node_ids = [int(l.split()[0]) for l in lines]
            assert node_ids == sorted(node_ids)
        finally:
            os.unlink(path)


class TestParseDFile:

    def test_parses_nodes(self, small_mesh_d):
        nodes, elements, elem_type, nod = parse_d_file(small_mesh_d)
        assert len(nodes) == 27

    def test_parses_elements(self, small_mesh_d):
        nodes, elements, elem_type, nod = parse_d_file(small_mesh_d)
        assert len(elements) == 8

    def test_element_type(self, small_mesh_d):
        nodes, elements, elem_type, nod = parse_d_file(small_mesh_d)
        assert elem_type == 'hexahedron'
        assert nod == 8


class TestAssignZones:

    def test_two_face_zones(self, small_mesh_d):
        nodes, elements, _, nod = parse_d_file(small_mesh_d)
        boundary = find_boundary_nodes(nodes, elements, nod)
        zones = [
            {'axis_min': 0.0, 'axis_max': 0.1, 'temperature': 500.0},
            {'axis_min': 0.9, 'axis_max': 1.0, 'temperature': 293.0},
        ]
        fixed = assign_zones(nodes, boundary, zones, 'z')
        temps = set(fixed.values())
        assert temps == {500.0, 293.0}

    def test_zone_node_count(self, small_mesh_d):
        nodes, elements, _, nod = parse_d_file(small_mesh_d)
        boundary = find_boundary_nodes(nodes, elements, nod)
        # z=0 face: 9 nodes at normalized z=0.0, z=-1 face: 9 nodes at normalized z=1.0
        zones = [
            {'axis_min': 0.0, 'axis_max': 0.01, 'temperature': 500.0},
            {'axis_min': 0.99, 'axis_max': 1.0, 'temperature': 293.0},
        ]
        fixed = assign_zones(nodes, boundary, zones, 'z')
        assert len(fixed) == 18
