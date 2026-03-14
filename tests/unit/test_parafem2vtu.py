"""Unit tests for the ParaFEM to VTU converter (parafem2vtu.py)."""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'galaxy', 'tools', 'gdps_parafem2vtu'))

from parafem2vtu import (
    parse_d_file,
    parse_ensi_scalar,
    parse_bnd_file,
    parse_fix_file,
    build_vtu_tree,
    write_vtu,
    VTK_CELL_TYPES,
)


class TestParseDFile:

    def test_parses_nodes(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        assert len(nodes) == 27

    def test_parses_elements(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        assert len(elements) == 8

    def test_nod_is_8(self, small_mesh_d):
        _, _, nod = parse_d_file(small_mesh_d)
        assert nod == 8

    def test_node_has_3_coords(self, small_mesh_d):
        nodes, _, _ = parse_d_file(small_mesh_d)
        for nid, coords in nodes.items():
            assert len(coords) == 3

    def test_elements_have_mat_id(self, small_mesh_d):
        _, elements, _ = parse_d_file(small_mesh_d)
        for eid, data in elements.items():
            assert 'mat_id' in data
            assert 'nodes' in data


class TestParseEnsiScalar:

    def test_parses_ndptl(self, small_ensi_ndptl):
        values = parse_ensi_scalar(small_ensi_ndptl)
        assert len(values) == 27

    def test_values_are_floats(self, small_ensi_ndptl):
        values = parse_ensi_scalar(small_ensi_ndptl)
        assert all(isinstance(v, float) for v in values)

    def test_values_in_range(self, small_ensi_ndptl):
        values = parse_ensi_scalar(small_ensi_ndptl)
        assert all(300.0 <= v <= 900.0 for v in values)


class TestParseBndFile:

    def test_parses_bnd(self, small_mesh_bnd):
        flags = parse_bnd_file(small_mesh_bnd, 27)
        assert len(flags) == 27
        # Original bnd from p12meshgen has boundary nodes flagged
        assert sum(1 for f in flags if f > 0) > 0

    def test_missing_file_returns_zeros(self):
        flags = parse_bnd_file('/nonexistent/path.bnd', 10)
        assert flags == [0.0] * 10


class TestParseFixFile:

    def test_parses_fix(self, small_fix):
        values = parse_fix_file(small_fix, 27)
        assert len(values) == 27
        non_zero = sum(1 for v in values if v != 0.0)
        assert non_zero == 18

    def test_missing_file_returns_zeros(self):
        values = parse_fix_file('/nonexistent/path.fix', 10)
        assert values == [0.0] * 10


class TestBuildVtuTree:

    def test_basic_vtu_structure(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        root = build_vtu_tree(nodes, elements, nod)
        assert root.tag == 'VTKFile'
        assert root.attrib['type'] == 'UnstructuredGrid'
        piece = root.find('.//Piece')
        assert piece.attrib['NumberOfPoints'] == '27'
        assert piece.attrib['NumberOfCells'] == '8'

    def test_has_points(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        root = build_vtu_tree(nodes, elements, nod)
        points = root.find('.//Points/DataArray')
        assert points is not None
        assert points.attrib['NumberOfComponents'] == '3'

    def test_has_cells(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        root = build_vtu_tree(nodes, elements, nod)
        conn = root.find('.//Cells/DataArray[@Name="connectivity"]')
        offsets = root.find('.//Cells/DataArray[@Name="offsets"]')
        types = root.find('.//Cells/DataArray[@Name="types"]')
        assert conn is not None
        assert offsets is not None
        assert types is not None

    def test_cell_type_is_hex(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        root = build_vtu_tree(nodes, elements, nod)
        types = root.find('.//Cells/DataArray[@Name="types"]')
        type_vals = [int(x) for x in types.text.strip().split()]
        assert all(t == VTK_CELL_TYPES[8] for t in type_vals)
        assert len(type_vals) == 8

    def test_point_data_included(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        point_data = {'Temperature': [float(i) for i in range(27)]}
        root = build_vtu_tree(nodes, elements, nod, point_data=point_data)
        pd = root.find('.//PointData/DataArray[@Name="Temperature"]')
        assert pd is not None
        values = [float(x) for x in pd.text.strip().split()]
        assert len(values) == 27

    def test_cell_data_included(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        cell_data = {'MaterialID': [1, 1, 2, 2, 1, 1, 2, 2]}
        root = build_vtu_tree(nodes, elements, nod, cell_data=cell_data)
        cd = root.find('.//CellData/DataArray[@Name="MaterialID"]')
        assert cd is not None
        values = [int(x) for x in cd.text.strip().split()]
        assert len(values) == 8


class TestWriteVtu:

    def test_writes_valid_xml(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        root = build_vtu_tree(nodes, elements, nod)
        with tempfile.NamedTemporaryFile(suffix='.vtu', delete=False) as f:
            path = f.name
        try:
            write_vtu(root, path)
            # Should parse as valid XML
            tree = ET.parse(path)
            assert tree.getroot().tag == 'VTKFile'
        finally:
            os.unlink(path)

    def test_roundtrip_with_data(self, small_mesh_d, small_ensi_ndptl, small_fix):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        potentials = parse_ensi_scalar(small_ensi_ndptl)
        fix_vals = parse_fix_file(small_fix, len(nodes))
        point_data = {'Potential': potentials, 'FixedFreedoms': fix_vals}
        root = build_vtu_tree(nodes, elements, nod, point_data=point_data)

        with tempfile.NamedTemporaryFile(suffix='.vtu', delete=False) as f:
            path = f.name
        try:
            write_vtu(root, path)
            tree = ET.parse(path)
            pot = tree.find('.//PointData/DataArray[@Name="Potential"]')
            assert pot is not None
            vals = [float(x) for x in pot.text.strip().split()]
            assert len(vals) == 27
            assert vals[0] == pytest.approx(potentials[0], rel=1e-4)
        finally:
            os.unlink(path)
