"""Unit tests for gdps_sensor_to_bcs.py."""

import csv
import importlib.util
import os
import sys
import tempfile
import types

import pytest

TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'galaxy', 'tools'))
TOOL_DIR = os.path.join(TOOLS_DIR, 'gdps_sensor_to_bcs')


def _load_fresh_module(name, path):
    """Load a module from a file path, bypassing sys.modules cache."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_common():
    """Load parafem_common bypassing any mock in sys.modules."""
    real_name = '_parafem_common_real'
    if real_name in sys.modules:
        return sys.modules[real_name]
    mod = _load_fresh_module(real_name, os.path.join(TOOLS_DIR, 'parafem_common.py'))
    sys.modules[real_name] = mod
    return mod


def _get_tool():
    """Load gdps_sensor_to_bcs with the real parafem_common in sys.modules."""
    # Temporarily install the real parafem_common so the tool can import it
    common = _get_common()
    prev = sys.modules.get('parafem_common')
    sys.modules['parafem_common'] = common
    if TOOLS_DIR not in sys.path:
        sys.path.insert(0, TOOLS_DIR)
    try:
        mod = _load_fresh_module('_gdps_sensor_to_bcs_fresh', os.path.join(TOOL_DIR, 'gdps_sensor_to_bcs.py'))
    finally:
        if prev is None:
            sys.modules.pop('parafem_common', None)
        else:
            sys.modules['parafem_common'] = prev
    return mod


ZONE_TEMPLATE = [
    {"col_name": "upstream_temp", "axis_min": 0.0, "axis_max": 0.1},
    {"col_name": "downstream_temp", "axis_min": 0.9, "axis_max": 1.0},
]


def _write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_bcs(path):
    with open(path) as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def test_basic_zone_mode(small_mesh_d):
    common = _get_common()
    tool = _get_tool()

    nodes, elements, _, nod = common.parse_d_file(small_mesh_d)
    boundary_nodes = common.find_boundary_nodes(nodes, elements, nod)

    csv_rows = [
        {"elapsed_seconds": "10.0", "upstream_temp": "600.0", "downstream_temp": "293.0"},
        {"elapsed_seconds": "20.0", "upstream_temp": "650.0", "downstream_temp": "293.0"},
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bcs', delete=False) as f:
        bcs_path = f.name

    try:
        _write_csv(csv_rows, csv_path)
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        tool.write_bcs_from_sensor(
            bcs_path, rows, ZONE_TEMPLATE, nodes, boundary_nodes,
            'zone', 'z', {}, dtim=10.0,
        )

        lines = _read_bcs(bcs_path)

        step_indices = [i for i, l in enumerate(lines) if l.strip().isdigit()]
        assert len(step_indices) == 2
        assert lines[step_indices[0]] == '1'
        assert lines[step_indices[1]] == '2'

        step1_rows = lines[step_indices[0] + 1: step_indices[1]]
        node_ids_1 = [int(r.split()[0]) for r in step1_rows]
        temps_1 = [float(r.split()[2]) for r in step1_rows]
        assert node_ids_1 == sorted(node_ids_1)
        assert len(node_ids_1) > 0
        assert any(abs(t - 600.0) < 1e-6 for t in temps_1)

        step2_rows = lines[step_indices[1] + 1:]
        node_ids_2 = [int(r.split()[0]) for r in step2_rows]
        temps_2 = [float(r.split()[2]) for r in step2_rows]
        assert node_ids_2 == sorted(node_ids_2)
        assert any(abs(t - 650.0) < 1e-6 for t in temps_2)

    finally:
        for p in (csv_path, bcs_path):
            if os.path.exists(p):
                os.unlink(p)


def test_node_order_matches_fix(small_mesh_d):
    common = _get_common()
    tool = _get_tool()

    nodes, elements, _, nod = common.parse_d_file(small_mesh_d)
    boundary_nodes = common.find_boundary_nodes(nodes, elements, nod)

    zones_for_assign = [
        {"axis_min": 0.0, "axis_max": 0.1, "temperature": 600.0},
        {"axis_min": 0.9, "axis_max": 1.0, "temperature": 293.0},
    ]
    expected_fixed = common.assign_zones(nodes, boundary_nodes, zones_for_assign, 'z')
    expected_order = sorted(expected_fixed.keys())

    csv_rows = [
        {"elapsed_seconds": "10.0", "upstream_temp": "600.0", "downstream_temp": "293.0"},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bcs', delete=False) as f:
        bcs_path = f.name

    try:
        _write_csv(csv_rows, csv_path)
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        tool.write_bcs_from_sensor(
            bcs_path, rows, ZONE_TEMPLATE, nodes, boundary_nodes,
            'zone', 'z', {}, dtim=10.0,
        )

        lines = _read_bcs(bcs_path)
        data_lines = lines[1:]
        actual_order = [int(l.split()[0]) for l in data_lines]

        assert actual_order == expected_order

    finally:
        for p in (csv_path, bcs_path):
            if os.path.exists(p):
                os.unlink(p)


def test_step_zero_skipped(small_mesh_d):
    common = _get_common()
    tool = _get_tool()

    nodes, elements, _, nod = common.parse_d_file(small_mesh_d)
    boundary_nodes = common.find_boundary_nodes(nodes, elements, nod)

    csv_rows = [
        {"elapsed_seconds": "0.0", "upstream_temp": "600.0", "downstream_temp": "293.0"},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bcs', delete=False) as f:
        bcs_path = f.name

    try:
        _write_csv(csv_rows, csv_path)
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        tool.write_bcs_from_sensor(
            bcs_path, rows, ZONE_TEMPLATE, nodes, boundary_nodes,
            'zone', 'z', {}, dtim=10.0,
        )

        with open(bcs_path) as f:
            content = f.read()
        assert content == ""

    finally:
        for p in (csv_path, bcs_path):
            if os.path.exists(p):
                os.unlink(p)


def test_nset_mode(small_mesh_d, small_nset):
    common = _get_common()
    tool = _get_tool()

    nodes, elements, _, nod = common.parse_d_file(small_mesh_d)
    boundary_nodes = common.find_boundary_nodes(nodes, elements, nod)
    nsets = common.parse_nset_file(small_nset)

    nset_template = [
        {"col_name": "upstream_temp", "nset_name": "UPSTREAM"},
        {"col_name": "downstream_temp", "nset_name": "DOWNSTREAM"},
    ]
    csv_rows = [
        {"elapsed_seconds": "30.0", "upstream_temp": "800.0", "downstream_temp": "293.0"},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bcs', delete=False) as f:
        bcs_path = f.name
    try:
        _write_csv(csv_rows, csv_path)
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        tool.write_bcs_from_sensor(
            bcs_path, rows, nset_template, nodes, boundary_nodes,
            'nset', 'z', nsets, dtim=10.0,
        )
        lines = _read_bcs(bcs_path)
        assert lines[0] == '3'  # step = round(30.0 / 10.0)
        node_ids = [int(l.split()[0]) for l in lines[1:]]
        assert node_ids == sorted(node_ids)
        upstream_ids = sorted(nsets['UPSTREAM'])
        for nid in upstream_ids:
            match = [l for l in lines[1:] if int(l.split()[0]) == nid]
            assert len(match) == 1
            assert float(match[0].split()[2]) == pytest.approx(800.0)
    finally:
        for p in (csv_path, bcs_path):
            if os.path.exists(p):
                os.unlink(p)


def test_empty_csv(small_mesh_d):
    common = _get_common()
    tool = _get_tool()

    nodes, elements, _, nod = common.parse_d_file(small_mesh_d)
    boundary_nodes = common.find_boundary_nodes(nodes, elements, nod)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
        writer = csv.DictWriter(f, fieldnames=['elapsed_seconds', 'upstream_temp', 'downstream_temp'])
        writer.writeheader()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bcs', delete=False) as f:
        bcs_path = f.name

    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        tool.write_bcs_from_sensor(
            bcs_path, rows, ZONE_TEMPLATE, nodes, boundary_nodes,
            'zone', 'z', {}, dtim=10.0,
        )

        with open(bcs_path) as f:
            content = f.read()
        assert content == ""

    finally:
        for p in (csv_path, bcs_path):
            if os.path.exists(p):
                os.unlink(p)
