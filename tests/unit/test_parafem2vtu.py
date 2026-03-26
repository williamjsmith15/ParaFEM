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
    parse_ensi_vector,
    parse_bnd_file,
    parse_fix_file,
    build_vtu_tree,
    write_vtu,
    write_pvd,
    find_ensi_files,
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
        import math
        values = parse_fix_file(small_fix, 27)
        assert len(values) == 27
        fixed = sum(1 for v in values if not math.isnan(v))
        assert fixed == 18

    def test_missing_file_returns_nan(self):
        import math
        values = parse_fix_file('/nonexistent/path.fix', 10)
        assert len(values) == 10
        assert all(math.isnan(v) for v in values)


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


class TestParseEnsiVector:

    def test_parses_planar_layout(self):
        # 3 nodes, vector data stored as: x0 x1 x2 y0 y1 y2 z0 z1 z2
        content = (
            "header line 1\n"
            "header line 2\n"
            "header line 3\n"
            "header line 4\n"
            "1.0\n2.0\n3.0\n"
            "4.0\n5.0\n6.0\n"
            "7.0\n8.0\n9.0\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ensi', delete=False) as f:
            f.write(content)
            path = f.name
        try:
            result = parse_ensi_vector(path, nn=3)
            assert len(result) == 3
            assert result[0] == [1.0, 4.0, 7.0]
            assert result[1] == [2.0, 5.0, 8.0]
            assert result[2] == [3.0, 6.0, 9.0]
        finally:
            os.unlink(path)

    def test_returns_list_of_xyz(self):
        content = (
            "h1\nh2\nh3\nh4\n"
            "10.0\n20.0\n"
            "30.0\n40.0\n"
            "50.0\n60.0\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ensi', delete=False) as f:
            f.write(content)
            path = f.name
        try:
            result = parse_ensi_vector(path, nn=2)
            assert len(result) == 2
            for vec in result:
                assert len(vec) == 3
        finally:
            os.unlink(path)


class TestWritePvd:

    def test_writes_valid_pvd(self):
        entries = [(0.0, '/tmp/step_000000.vtu'), (500.0, '/tmp/step_000001.vtu')]
        with tempfile.NamedTemporaryFile(suffix='.pvd', delete=False) as f:
            pvd_path = f.name
        try:
            write_pvd(entries, pvd_path)
            tree = ET.parse(pvd_path)
            root = tree.getroot()
            assert root.tag == 'VTKFile'
            assert root.attrib['type'] == 'Collection'
            datasets = root.findall('.//DataSet')
            assert len(datasets) == 2
        finally:
            os.unlink(pvd_path)

    def test_timestep_attributes(self):
        entries = [(0.0, '/tmp/step_000000.vtu'), (500.0, '/tmp/step_000001.vtu')]
        with tempfile.NamedTemporaryFile(suffix='.pvd', delete=False) as f:
            pvd_path = f.name
        try:
            write_pvd(entries, pvd_path)
            tree = ET.parse(pvd_path)
            datasets = tree.findall('.//DataSet')
            assert datasets[0].attrib['timestep'] == '0.0'
            assert datasets[1].attrib['timestep'] == '500.0'
            assert datasets[0].attrib['file'] == 'step_000000.vtu'
            assert datasets[1].attrib['file'] == 'step_000001.vtu'
        finally:
            os.unlink(pvd_path)


class TestFindEnsiFiles:

    def test_finds_grouped_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ['job.ensi.NDTTR-000001', 'job.ensi.NDTTR-000002',
                         'job.ensi.NDPTL-000001']:
                open(os.path.join(tmpdir, name), 'w').close()
            result = find_ensi_files(tmpdir, 'job')
            assert 'NDTTR' in result
            assert 'NDPTL' in result
            assert len(result['NDTTR']) == 2
            assert len(result['NDPTL']) == 1

    def test_sorted_by_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ['job.ensi.NDTTR-000003', 'job.ensi.NDTTR-000001',
                         'job.ensi.NDTTR-000002']:
                open(os.path.join(tmpdir, name), 'w').close()
            result = find_ensi_files(tmpdir, 'job')
            steps = [s for s, _ in result['NDTTR']]
            assert steps == [1, 2, 3]


class TestBuildVtuTreeVector:

    def test_vector_point_data_has_3_components(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        nn = len(nodes)
        vectors = [[float(i), float(i + 1), float(i + 2)] for i in range(nn)]
        point_data = {'Displacement': vectors}
        root = build_vtu_tree(nodes, elements, nod, point_data=point_data)
        da = root.find('.//PointData/DataArray[@Name="Displacement"]')
        assert da is not None
        assert da.attrib['NumberOfComponents'] == '3'

    def test_vector_values_roundtrip(self, small_mesh_d):
        nodes, elements, nod = parse_d_file(small_mesh_d)
        nn = len(nodes)
        vectors = [[1.0, 2.0, 3.0]] * nn
        point_data = {'Displacement': vectors}
        root = build_vtu_tree(nodes, elements, nod, point_data=point_data)
        da = root.find('.//PointData/DataArray[@Name="Displacement"]')
        lines = [l.strip() for l in da.text.strip().split('\n') if l.strip()]
        assert len(lines) == nn


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
