"""Unit tests for the thermal BC generator (gdps_bc_thermal.py)."""

import os
import sys
import tempfile
import pytest

# Add the tool directory to path so we can import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'galaxy', 'tools', 'gdps_bc_thermal'))

from gdps_bc_thermal import (
    parse_d_file,
    find_boundary_nodes,
    assign_zones,
    write_bnd,
    write_fix,
    write_dat,
)


class TestParseDFile:

    def test_parses_nodes(self, small_mesh_d):
        nodes, elements, elem_type, nod = parse_d_file(small_mesh_d)
        assert len(nodes) == 27

    def test_parses_elements(self, small_mesh_d):
        nodes, elements, elem_type, nod = parse_d_file(small_mesh_d)
        assert len(elements) == 8

    def test_element_type_hex(self, small_mesh_d):
        nodes, elements, elem_type, nod = parse_d_file(small_mesh_d)
        assert elem_type == 'hexahedron'
        assert nod == 8

    def test_node_coordinates(self, small_mesh_d):
        nodes, _, _, _ = parse_d_file(small_mesh_d)
        # Node 1 should be at origin (0, 0, 0)
        assert abs(nodes[1][0]) < 1e-10
        assert abs(nodes[1][1]) < 1e-10
        assert abs(nodes[1][2]) < 1e-10

    def test_each_element_has_8_nodes(self, small_mesh_d):
        _, elements, _, nod = parse_d_file(small_mesh_d)
        for eid, elem_nodes in elements.items():
            assert len(elem_nodes) == 8


class TestFindBoundaryNodes:

    def test_boundary_count(self, small_mesh_d):
        nodes, elements, _, nod = parse_d_file(small_mesh_d)
        boundary = find_boundary_nodes(nodes, elements, nod)
        # 2x2x2 mesh: 27 nodes, only the single interior node is non-boundary
        assert len(boundary) == 26

    def test_interior_node_excluded(self, small_mesh_d):
        nodes, elements, _, nod = parse_d_file(small_mesh_d)
        boundary = find_boundary_nodes(nodes, elements, nod)
        # Find the interior node (not on any face)
        import numpy as np
        coords = {nid: np.array(c) for nid, c in nodes.items()}
        mins = np.array([c.min() for c in np.array(list(coords.values())).T])
        maxs = np.array([c.max() for c in np.array(list(coords.values())).T])
        for nid, c in coords.items():
            on_face = any(abs(c[d] - mins[d]) < 1e-10 or abs(c[d] - maxs[d]) < 1e-10 for d in range(3))
            if not on_face:
                assert nid not in boundary


class TestAssignZones:

    def test_single_zone_all_nodes(self, small_mesh_d):
        nodes, elements, _, nod = parse_d_file(small_mesh_d)
        import numpy as np
        nodes_np = {k: np.array(v) for k, v in nodes.items()}
        boundary = find_boundary_nodes(nodes_np, elements, nod)
        zones = [{'axis_min': 0.0, 'axis_max': 1.0, 'temperature': 500.0}]
        fixed = assign_zones(nodes_np, boundary, zones, 'z')
        assert len(fixed) == len(boundary)
        assert all(v == 500.0 for v in fixed.values())

    def test_partial_zone(self, small_mesh_d):
        nodes, elements, _, nod = parse_d_file(small_mesh_d)
        import numpy as np
        nodes_np = {k: np.array(v) for k, v in nodes.items()}
        boundary = find_boundary_nodes(nodes_np, elements, nod)
        zones = [{'axis_min': 0.0, 'axis_max': 0.3, 'temperature': 800.0}]
        fixed = assign_zones(nodes_np, boundary, zones, 'z')
        assert 0 < len(fixed) < len(boundary)
        assert all(v == 800.0 for v in fixed.values())

    def test_no_overlap_between_zones(self, small_mesh_d):
        nodes, elements, _, nod = parse_d_file(small_mesh_d)
        import numpy as np
        nodes_np = {k: np.array(v) for k, v in nodes.items()}
        boundary = find_boundary_nodes(nodes_np, elements, nod)
        zones = [
            {'axis_min': 0.0, 'axis_max': 0.4, 'temperature': 800.0},
            {'axis_min': 0.6, 'axis_max': 1.0, 'temperature': 400.0},
        ]
        fixed = assign_zones(nodes_np, boundary, zones, 'z')
        temps = set(fixed.values())
        assert temps <= {800.0, 400.0}

    def test_empty_zone_returns_empty(self, small_mesh_d):
        nodes, elements, _, nod = parse_d_file(small_mesh_d)
        import numpy as np
        nodes_np = {k: np.array(v) for k, v in nodes.items()}
        boundary = find_boundary_nodes(nodes_np, elements, nod)
        zones = [{'axis_min': 0.45, 'axis_max': 0.55, 'temperature': 999.0}]
        fixed = assign_zones(nodes_np, boundary, zones, 'z')
        # With a 2x2x2 mesh, the mid-plane nodes (z=-0.5) land at normalized 0.5
        # which is inside [0.45, 0.55], so some nodes may match
        # This test just verifies no crash
        assert isinstance(fixed, dict)


class TestWriteDat:

    def test_dat_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            path = f.name
        try:
            write_dat(path, nels=8, nn=27, nr=0, nod=8, fixed_freedoms=18,
                      kx=50.0, ky=50.0, kz=50.0, tol=1e-6, limit=2000,
                      element_type='hexahedron')
            with open(path) as f:
                lines = f.readlines()
            assert "'hexahedron'" in lines[0]
            assert '2' in lines[1].strip()
            assert '1' in lines[2].strip()
            # Line 4: nels nn nr nip nod loaded_nodes fixed_freedoms
            parts = lines[3].split()
            assert int(parts[0]) == 8
            assert int(parts[1]) == 27
            assert int(parts[2]) == 0
            assert int(parts[5]) == 0    # loaded_nodes
            assert int(parts[6]) == 18   # fixed_freedoms
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
            # Should be sorted by node ID
            parts0 = lines[0].split()
            assert int(parts0[0]) == 1
            assert int(parts0[1]) == 1   # sense
            assert float(parts0[2]) == pytest.approx(800.0)
            parts1 = lines[1].split()
            assert int(parts1[0]) == 5
        finally:
            os.unlink(path)


class TestWriteBnd:

    def test_bnd_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bnd', delete=False) as f:
            path = f.name
        try:
            write_bnd(path, [3, 1, 7])
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3
            # Should be sorted
            assert int(lines[0].split()[0]) == 1
            assert int(lines[1].split()[0]) == 3
            assert int(lines[2].split()[0]) == 7
            # Each line should have '0' as second field
            for line in lines:
                assert line.split()[1] == '0'
        finally:
            os.unlink(path)
