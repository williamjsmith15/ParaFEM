"""Unit tests for the BC schedule generator (gdps_bc_schedule.py)."""

import os
import sys
import tempfile
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'galaxy', 'tools', 'gdps_bc_schedule'))
from gdps_bc_schedule import write_bcs, main as bcs_main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'galaxy', 'tools'))
from parafem_common import parse_d_file, find_boundary_nodes, parse_nset_file

def test_bcs_format_and_node_order(small_mesh_d):
    nodes, elements, _, nod = parse_d_file(small_mesh_d)
    boundary_nodes = find_boundary_nodes(nodes, elements, nod)
    schedule = [
        {"step": 5, "zones": [{"axis_min": 0.0, "axis_max": 0.1, "temperature": 555.0}]},
        {"step": 10, "zones": [{"axis_min": 0.0, "axis_max": 0.1, "temperature": 999.0}]}
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.bcs', delete=False) as f:
        path = f.name

    try:
        write_bcs(path, schedule, nodes, boundary_nodes, 'zone', 'z', {})
        with open(path) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        assert lines[0] == '5'
        # 9 nodes on the z=0 face for small_2x2x2 mesh
        step1_nodes = [int(line.split()[0]) for line in lines[1:10]]
        step1_temps = [float(line.split()[2]) for line in lines[1:10]]
        assert len(step1_nodes) == 9
        assert step1_nodes == sorted(step1_nodes)
        assert all(temp == pytest.approx(555.0) for temp in step1_temps)

        assert lines[10] == '10'
        step2_nodes = [int(line.split()[0]) for line in lines[11:20]]
        step2_temps = [float(line.split()[2]) for line in lines[11:20]]
        assert len(step2_nodes) == 9
        assert step2_nodes == sorted(step2_nodes)
        assert all(temp == pytest.approx(999.0) for temp in step2_temps)

    finally:
        if os.path.exists(path):
            os.unlink(path)

def test_empty_schedule(small_mesh_d):
    nodes, elements, _, nod = parse_d_file(small_mesh_d)
    boundary_nodes = find_boundary_nodes(nodes, elements, nod)
    schedule = []

    with tempfile.NamedTemporaryFile(mode='w', suffix='.bcs', delete=False) as f:
        path = f.name

    try:
        write_bcs(path, schedule, nodes, boundary_nodes, 'zone', 'z', {})
        with open(path) as f:
            content = f.read()
        assert content == ""
    finally:
        os.unlink(path)

def test_main_script_zone_mode(small_mesh_d, monkeypatch):
    schedule_data = [
        {"step": 1, "zones": [{"axis_min": 0.9, "axis_max": 1.0, "temperature": 300.0}]}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as sched_f:
        json.dump(schedule_data, sched_f)
        schedule_path = sched_f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.bcs', delete=False) as out_f:
        output_path = out_f.name
    
    try:
        args = [
            'gdps_bc_schedule.py',
            '--mesh_d', small_mesh_d,
            '--schedule_config', schedule_path,
            '--bc_mode', 'zone',
            '--bc_axis', 'z',
            '--output_bcs', output_path,
        ]
        monkeypatch.setattr(sys, 'argv', args)
        bcs_main()
        
        with open(output_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 10 # 1 step header + 9 node lines
        assert lines[0].strip() == '1'
        assert float(lines[1].split()[2]) == pytest.approx(300.0)
        
    finally:
        os.unlink(schedule_path)
        os.unlink(output_path)
