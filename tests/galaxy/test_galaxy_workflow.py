"""Galaxy integration tests: run the steady-thermal workflow end-to-end.

Requires:
  - Galaxy running at localhost:8080 (or GALAXY_URL env var)
  - bioblend installed (pip install bioblend)
  - Bootstrap completed (workflow imported, tools loaded)

Run with: pytest tests/galaxy/ -v --tb=short
"""

import io
import re
import tarfile
import time
import xml.etree.ElementTree as ET

import pytest


EXPECTED_TOOLS = [
    "gdps_mesh_convert",
    "gdps_meshgen_box",
    "gdps_mesh_import",
    "gdps_mesh_preview",
    "gdps_bc_thermal",
    "gdps_steady_thermal",
    "gdps_postprocess",
    "gdps_parafem2vtu",
    "gdps_bc_transient",
    "gdps_transient_thermal",
    "gdps_parafem2vtu_transient",
    "gdps_extract_ic",
    "gdps_bc_schedule",
    "gdps_sensor_to_bcs",
    "gdps_sievert_bc",
    "gdps_compute_diffusivity",
    "gdps_rfemcube",
    "gdps_rfembc_thermal",
    "gdps_rfemfield_thermal",
    "gdps_rfemsolve_thermal",
    "gdps_rfemmc_runner",
    "gdps_rfemmc_collect",
    "gdps_rfem_map",
]

WORKFLOW_TIMEOUT = 600  # 10 minutes max for full pipeline
POLL_INTERVAL = 10


def _wait_for_workflow(gi, invocation_id, history_id, timeout, label="Workflow"):
    """Poll until all datasets are done, failing immediately on any error dataset."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        inv = gi.invocations.show_invocation(invocation_id)
        state = inv["state"]
        if state == "scheduled":
            datasets = gi.histories.show_history(history_id, contents=True)
            if datasets:
                failed = [ds for ds in datasets if ds["state"] == "error"]
                if failed:
                    _dump_errors(gi, history_id)
                    pytest.fail(
                        f"{label}: {len(failed)} dataset(s) in error state: "
                        f"{[d['name'] for d in failed]}"
                    )
                if all(ds["state"] in ("ok", "deleted", "discarded") for ds in datasets):
                    break
        elif state in ("cancelled", "failed"):
            _dump_errors(gi, history_id)
            pytest.fail(f"{label} invocation {state}")
        time.sleep(POLL_INTERVAL)
    else:
        _dump_errors(gi, history_id)
        pytest.fail(f"{label} timed out after {timeout}s")


class TestGalaxySetup:

    def test_galaxy_reachable(self, gi):
        version = gi.config.get_version()
        assert "version_major" in version

    def test_tools_loaded(self, gi):
        tools = gi.tools.get_tools()
        tool_ids = {t["id"] for t in tools}
        missing = [t for t in EXPECTED_TOOLS if t not in tool_ids]
        assert not missing, f"Missing tools: {missing}"

    def test_workflow_imported(self, gi, workflow_id):
        wf = gi.workflows.show_workflow(workflow_id)
        assert wf["name"] == "Steady-State Thermal (ParaFEM p123)"
        assert len(wf["steps"]) == 15  # 10 parameter inputs + 5 tool steps


@pytest.fixture(scope="module")
def workflow_result(gi, workflow_id):
    """Run the workflow once and return (gi, history_id) for all tests."""
    history = gi.histories.create_history(name="test-steady-thermal")
    history_id = history["id"]

    invocation = gi.workflows.invoke_workflow(
        workflow_id,
        history_id=history_id,
        allow_tool_state_corrections=True,
    )
    invocation_id = invocation["id"]

    _wait_for_workflow(gi, invocation_id, history_id, WORKFLOW_TIMEOUT, "Steady-thermal workflow")

    yield gi, history_id

    gi.histories.delete_history(history_id, purge=True)


def _dump_errors(gi, history_id):
    datasets = gi.histories.show_history(history_id, contents=True)
    for ds in datasets:
        if ds["state"] == "error":
            print(f"\n--- ERROR: {ds['name']} ---")
            try:
                detail = gi.datasets.show_dataset(ds["id"])
                job_id = detail.get("creating_job")
                if job_id:
                    job = gi.jobs.show_job(job_id, full_details=True)
                    print(f"  stdout: {job.get('stdout', '')[:500]}")
                    print(f"  stderr: {job.get('stderr', '')[:500]}")
            except Exception as e:
                print(f"  (could not fetch job details: {e})")


def _download_dataset(gi, history_id, name_pattern, occurrence=1):
    """Download a dataset by name pattern. occurrence=N picks the Nth match (1-indexed)."""
    datasets = gi.histories.show_history(history_id, contents=True)
    count = 0
    for ds in datasets:
        if re.search(name_pattern, ds["name"], re.IGNORECASE):
            count += 1
            if count == occurrence:
                return gi.datasets.download_dataset(ds["id"])
    available = [ds["name"] for ds in datasets]
    pytest.fail(
        f"No dataset matching '{name_pattern}' (occurrence {occurrence}). Available: {available}"
    )


def _download_text(gi, history_id, name_pattern, occurrence=1):
    data = _download_dataset(gi, history_id, name_pattern, occurrence=occurrence)
    return data.decode("utf-8", errors="replace")


class TestSteadyThermalWorkflow:

    def test_solver_converges(self, workflow_result):
        gi, history_id = workflow_result
        res = _download_text(gi, history_id, r"Results summary.*\.res")
        match = re.search(r"iterations to convergence was\s+(\d+)", res)
        assert match, f"No convergence info in .res output:\n{res[:500]}"
        iters = int(match.group(1))
        assert iters < 200, f"Solver took {iters} iterations"

    def test_no_nan_in_results(self, workflow_result):
        gi, history_id = workflow_result
        res = _download_text(gi, history_id, r"Results summary.*\.res")
        assert "NaN" not in res, "NaN found in solver results"

    # --- VTU output tests ---

    def test_vtu_valid(self, workflow_result):
        gi, history_id = workflow_result
        vtu_bytes = _download_text(gi, history_id, r"VTK output.*\.vtu")
        tree = ET.fromstring(vtu_bytes)
        assert tree.tag == "VTKFile"
        piece = tree.find(".//Piece")
        assert piece is not None
        num_points = int(piece.attrib["NumberOfPoints"])
        assert num_points > 0, "VTU has no points"

    def test_vtu_has_temperature_field(self, workflow_result):
        gi, history_id = workflow_result
        vtu_bytes = _download_text(gi, history_id, r"VTK output.*\.vtu")
        tree = ET.fromstring(vtu_bytes)
        potential = tree.find('.//PointData/DataArray[@Name="Potential"]')
        assert potential is not None, "No Potential field in VTU"
        vals = [float(x) for x in potential.text.strip().split()]
        assert len(vals) > 0
        assert all(not (v != v) for v in vals), "NaN in VTU temperature field"

    def test_vtu_temperature_range(self, workflow_result):
        gi, history_id = workflow_result
        vtu_bytes = _download_text(gi, history_id, r"VTK output.*\.vtu")
        tree = ET.fromstring(vtu_bytes)
        potential = tree.find('.//PointData/DataArray[@Name="Potential"]')
        vals = [float(x) for x in potential.text.strip().split()]
        assert all(300 <= v <= 900 for v in vals), \
            f"Temperatures outside expected range: min={min(vals):.1f}, max={max(vals):.1f}"

    # --- EnSight output tests ---

    def test_ensi_case_file_valid(self, workflow_result):
        gi, history_id = workflow_result
        case_text = _download_text(gi, history_id, r"EnSight case file.*\.case")
        assert "FORMAT" in case_text, "Missing FORMAT section in .case file"
        assert "GEOMETRY" in case_text, "Missing GEOMETRY section in .case file"
        assert "job.ensi.geo" in case_text, "Missing geometry reference in .case"

    def test_ensi_case_has_variable(self, workflow_result):
        gi, history_id = workflow_result
        case_text = _download_text(gi, history_id, r"EnSight case file.*\.case")
        assert "VARIABLE" in case_text, "Missing VARIABLE section in .case file"
        assert "Potential" in case_text, \
            f"No Potential variable in .case file:\n{case_text}"

    def test_ensi_tarball_contents(self, workflow_result):
        gi, history_id = workflow_result
        raw = _download_dataset(gi, history_id, r"EnSight Gold package.*\.tar\.gz")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        names = tf.getnames()
        tf.close()

        assert any("ensi.case" in n for n in names), \
            f"No .ensi.case in tarball. Contents: {names}"
        assert any("ensi.geo" in n for n in names), \
            f"No .ensi.geo in tarball. Contents: {names}"
        assert any("NDPTL" in n for n in names), \
            f"No NDPTL variable file in tarball. Contents: {names}"

    def test_ensi_geo_has_nodes(self, workflow_result):
        gi, history_id = workflow_result
        raw = _download_dataset(gi, history_id, r"EnSight Gold package.*\.tar\.gz")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        geo_member = None
        for m in tf.getmembers():
            if "ensi.geo" in m.name:
                geo_member = m
                break
        assert geo_member is not None, "No .ensi.geo in tarball"
        geo_text = tf.extractfile(geo_member).read().decode("utf-8", errors="replace")
        tf.close()
        assert "coordinates" in geo_text.lower(), \
            f"No coordinates section in .ensi.geo"


TRANSIENT_WORKFLOW_TIMEOUT = 300


@pytest.fixture(scope="module")
def transient_workflow_result(gi, transient_workflow_id):
    """Run the transient workflow once and return (gi, history_id) for all tests."""
    history = gi.histories.create_history(name="test-transient-thermal")
    history_id = history["id"]

    invocation = gi.workflows.invoke_workflow(
        transient_workflow_id,
        history_id=history_id,
        allow_tool_state_corrections=True,
    )
    invocation_id = invocation["id"]

    _wait_for_workflow(gi, invocation_id, history_id, TRANSIENT_WORKFLOW_TIMEOUT, "Transient thermal workflow")

    yield gi, history_id

    gi.histories.delete_history(history_id, purge=True)


class TestTransientThermalWorkflow:

    # --- Workflow metadata ---

    def test_workflow_imported(self, gi, transient_workflow_id):
        wf = gi.workflows.show_workflow(transient_workflow_id)
        assert wf["name"] == "Transient Thermal (ParaFEM p124)"

    def test_workflow_has_correct_steps(self, gi, transient_workflow_id):
        wf = gi.workflows.show_workflow(transient_workflow_id)
        assert len(wf["steps"]) == 20, (
            f"Expected 20 steps (16 params + 4 tools), got {len(wf['steps'])}"
        )

    # --- Stage 1: Mesh generation ---

    def test_mesh_d_has_structure(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        # Use the bc pass-through label which is unique
        text = _download_text(gi, history_id, r"Mesh geometry \(\.d\) \[pass-through\]")
        assert "*NODES" in text
        assert "*ELEMENTS" in text

    def test_mesh_d_node_count(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        text = _download_text(gi, history_id, r"Mesh geometry \(\.d\) \[pass-through\]")
        # Default nxe=nye=nze=5: (5+1)^3 = 216 nodes
        nodes_section = text.split("*ELEMENTS")[0].split("*NODES")[1]
        node_lines = [l for l in nodes_section.strip().split("\n") if l.strip()]
        assert len(node_lines) == 216, f"Expected 216 nodes (5x5x5 mesh), got {len(node_lines)}"

    def test_mesh_bnd_has_surface_nodes(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        text = _download_text(gi, history_id, r"Boundary nodes \(\.bnd\)")
        lines = [l for l in text.strip().split("\n") if l.strip()]
        assert len(lines) > 0, "No boundary nodes in .bnd"
        for line in lines[:5]:
            parts = line.split()
            assert len(parts) == 2, f"Expected 2 cols in .bnd: {line}"

    # --- Stage 2: BC + material generation ---

    def test_dat_has_time_parameters(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        text = _download_text(gi, history_id, r"Control data \(\.dat\)")
        lines = text.strip().split("\n")
        # Line 5 (index 4): val0 dtim nstep npri
        time_line = lines[4].split()
        assert float(time_line[0]) == pytest.approx(293.0), "val0 should be 293K"
        assert float(time_line[1]) == pytest.approx(500.0), "dtim should be 500s"
        assert int(time_line[2]) == 100, "nstep should be 100"
        assert int(time_line[3]) == 10, "npri should be 10"

    def test_mat_has_material_values(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        text = _download_text(gi, history_id, r"Material properties \(\.mat\)")
        assert "*MATERIAL" in text
        parts = text.strip().split("\n")[2].split()
        assert float(parts[1]) == pytest.approx(50.0), "kx should be 50 W/m.K"
        assert float(parts[4]) == pytest.approx(7800.0), "rho should be 7800 kg/m3"
        assert float(parts[5]) == pytest.approx(500.0), "cp should be 500 J/kg.K"

    def test_fix_has_prescribed_temperatures(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        text = _download_text(gi, history_id, r"Fixed freedoms \(\.fix\)")
        lines = [l for l in text.strip().split("\n") if l.strip()]
        assert len(lines) > 0, "No fixed freedoms generated"
        temps = set()
        for line in lines:
            parts = line.split()
            assert len(parts) == 3, f"Expected 3 cols in .fix: {line}"
            temps.add(float(parts[2]))
        # Both boundary zone temperatures (500K and 293K) should appear
        assert 500.0 in temps, "500K zone not found in .fix"
        assert 293.0 in temps, "293K zone not found in .fix"

    # --- Stage 3: Transient solver ---

    def test_res_no_nan(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        text = _download_text(gi, history_id, r"Results summary \(\.res\)")
        assert "NaN" not in text

    def test_res_has_timestep_rows(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        text = _download_text(gi, history_id, r"Results summary \(\.res\)")
        # p124 writes t=0 initial row + nstep/npri rows = 11 total (nstep=100, npri=10)
        rows = re.findall(r'^\s+([\d.E+\-]+)\s+([\d.E+\-]+)', text, re.MULTILINE)
        assert len(rows) == 11, f"Expected 11 time output rows (t=0 + nstep/npri), got {len(rows)}"

    def test_res_temperatures_in_range(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        text = _download_text(gi, history_id, r"Results summary \(\.res\)")
        rows = re.findall(r'^\s+([\d.E+\-]+)\s+([\d.E+\-]+)', text, re.MULTILINE)
        temps = [float(r[1]) for r in rows]
        assert all(not (t != t) for t in temps), "NaN in .res temperature column"
        # nres=1 is a prescribed BC node; all temps should be within physical bounds
        assert all(285 <= t <= 510 for t in temps), \
            f"Monitor node temperatures outside BC range: {temps}"

    def test_ensi_tarball_has_ndttr_files(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        raw = _download_dataset(gi, history_id, r"EnSight output")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        names = tf.getnames()
        tf.close()
        ndttr = [n for n in names if "NDTTR" in n]
        # p124 writes initial state (NDTTR-000000) + nstep/npri = 11 files total
        assert len(ndttr) == 11, f"Expected 11 NDTTR files (t=0 + nstep/npri), got {len(ndttr)}: {names}"

    def test_ensi_temperatures_in_range(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        raw = _download_dataset(gi, history_id, r"EnSight output")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        members = sorted([m for m in tf.getmembers() if "NDTTR" in m.name], key=lambda m: m.name)
        text = tf.extractfile(members[-1]).read().decode("utf-8", errors="replace")
        tf.close()
        lines = text.split("\n")
        data_start = next(i for i, l in enumerate(lines) if "coordinates" in l) + 1
        temps = [float(l.strip()) for l in lines[data_start:] if l.strip()]
        assert len(temps) == 216, f"Expected 216 node values, got {len(temps)}"
        assert all(not (t != t) for t in temps), "NaN in NDTTR file"
        assert all(285 <= t <= 510 for t in temps), \
            f"Temperatures outside expected range: min={min(temps):.1f}, max={max(temps):.1f}"

    # --- Stage 4: VTU time series ---

    def test_vtu_tarball_has_pvd_and_vtu(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        raw = _download_dataset(gi, history_id, r"VTK time series")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        names = tf.getnames()
        tf.close()
        assert any(n.endswith(".pvd") for n in names), f"No .pvd in tarball: {names}"
        assert any(n.endswith(".vtu") for n in names), f"No .vtu in tarball: {names}"

    def test_pvd_has_correct_timesteps(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        raw = _download_dataset(gi, history_id, r"VTK time series")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        pvd_member = next(m for m in tf.getmembers() if m.name.endswith(".pvd"))
        pvd_text = tf.extractfile(pvd_member).read().decode("utf-8", errors="replace")
        tf.close()
        root = ET.fromstring(pvd_text)
        assert root.tag == "VTKFile"
        assert root.attrib.get("type") == "Collection"
        datasets = root.findall(".//DataSet")
        # p124 writes initial state + nstep/npri outputs = 11 total
        assert len(datasets) == 11, f"Expected 11 timesteps in PVD, got {len(datasets)}"
        times = [float(d.attrib["timestep"]) for d in datasets]
        assert times == sorted(times), "PVD timesteps not in ascending order"
        assert times[0] == pytest.approx(0.0), f"First timestep should be 0s (initial state), got {times[0]}"
        assert times[-1] == pytest.approx(50000.0), f"Last timestep should be 50000s (dtim*nstep), got {times[-1]}"

    def test_vtu_has_temperature_field(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        raw = _download_dataset(gi, history_id, r"VTK time series")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        vtu_members = sorted([m for m in tf.getmembers() if m.name.endswith(".vtu")], key=lambda m: m.name)
        vtu_text = tf.extractfile(vtu_members[-1]).read().decode("utf-8", errors="replace")
        tf.close()
        root = ET.fromstring(vtu_text)
        piece = root.find(".//Piece")
        assert piece is not None
        assert int(piece.attrib["NumberOfPoints"]) == 216
        temp = root.find('.//PointData/DataArray[@Name="Temperature"]')
        assert temp is not None, "No Temperature field in VTU"
        vals = [float(x) for x in temp.text.strip().split()]
        assert len(vals) == 216
        assert all(not (v != v) for v in vals), "NaN in VTU Temperature field"
        assert all(285 <= v <= 510 for v in vals), \
            f"VTU temperatures outside range: min={min(vals):.1f}, max={max(vals):.1f}"

    def test_vtu_has_fixed_freedoms_field(self, transient_workflow_result):
        gi, history_id = transient_workflow_result
        raw = _download_dataset(gi, history_id, r"VTK time series")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        vtu_members = sorted([m for m in tf.getmembers() if m.name.endswith(".vtu")], key=lambda m: m.name)
        vtu_text = tf.extractfile(vtu_members[0]).read().decode("utf-8", errors="replace")
        tf.close()
        root = ET.fromstring(vtu_text)
        ff = root.find('.//PointData/DataArray[@Name="FixedFreedoms"]')
        assert ff is not None, "No FixedFreedoms field in VTU"
        vals = [float(x) for x in ff.text.strip().split()]
        # BC nodes have prescribed temp, free nodes are NaN
        non_nan = [v for v in vals if v == v]
        assert len(non_nan) > 0, "No prescribed nodes in FixedFreedoms field"


IC_CHAIN_TIMEOUT = 600


@pytest.fixture(scope="module")
def ic_chain_result(gi, ic_chain_workflow_id):
    """Run the IC chain workflow once and return (gi, history_id) for all tests."""
    history = gi.histories.create_history(name="test-ic-chain")
    history_id = history["id"]

    invocation = gi.workflows.invoke_workflow(
        ic_chain_workflow_id,
        history_id=history_id,
        allow_tool_state_corrections=True,
    )
    invocation_id = invocation["id"]

    _wait_for_workflow(gi, invocation_id, history_id, IC_CHAIN_TIMEOUT, "IC chain workflow")

    yield gi, history_id

    gi.histories.delete_history(history_id, purge=True)


class TestICChainWorkflow:

    def test_workflow_imported(self, gi, ic_chain_workflow_id):
        wf = gi.workflows.show_workflow(ic_chain_workflow_id)
        assert wf["name"] == "Transient IC Chain (ParaFEM)"

    def test_workflow_has_correct_steps(self, gi, ic_chain_workflow_id):
        wf = gi.workflows.show_workflow(ic_chain_workflow_id)
        assert len(wf["steps"]) == 21, (
            f"Expected 21 steps (16 params + 5 tools), got {len(wf['steps'])}"
        )

    def test_run1_no_nan(self, ic_chain_result):
        gi, history_id = ic_chain_result
        text = _download_text(gi, history_id, r"Results summary", occurrence=1)
        assert "NaN" not in text

    def test_run1_has_timestep_rows(self, ic_chain_result):
        gi, history_id = ic_chain_result
        text = _download_text(gi, history_id, r"Results summary", occurrence=1)
        # nstep=10, npri=10: initial row + 1 output = 2 rows
        rows = re.findall(r'^\s+([\d.E+\-]+)\s+([\d.E+\-]+)', text, re.MULTILINE)
        assert len(rows) == 2, f"Expected 2 time output rows (t=0 + nstep/npri), got {len(rows)}"

    def test_extracted_ic_has_correct_node_count(self, ic_chain_result):
        gi, history_id = ic_chain_result
        text = _download_text(gi, history_id, r"initial conditions")
        lines = [l for l in text.strip().split("\n") if l.strip()]
        # First line is node count, rest are values: (3+1)^3 = 64 nodes
        node_count = int(lines[0])
        assert node_count == 64, f"Expected 64 nodes (3x3x3 mesh), got {node_count}"
        assert len(lines) - 1 == node_count, (
            f"Node count mismatch: header says {node_count}, got {len(lines)-1} values"
        )

    def test_extracted_ic_values_in_range(self, ic_chain_result):
        gi, history_id = ic_chain_result
        text = _download_text(gi, history_id, r"initial conditions")
        lines = [l for l in text.strip().split("\n") if l.strip()]
        values = [float(l) for l in lines[1:]]
        assert all(not (v != v) for v in values), "NaN in extracted IC values"
        assert all(285 <= v <= 510 for v in values), (
            f"IC values outside expected BC range: min={min(values):.1f}, max={max(values):.1f}"
        )

    def test_run2_no_nan(self, ic_chain_result):
        gi, history_id = ic_chain_result
        text = _download_text(gi, history_id, r"Results summary", occurrence=2)
        assert "NaN" not in text

    def test_run2_starts_from_run1_final_state(self, ic_chain_result):
        """Run 2's initial temperature (t=0 row) should match run 1's final temperature."""
        gi, history_id = ic_chain_result
        run1_text = _download_text(gi, history_id, r"Results summary", occurrence=1)
        run2_text = _download_text(gi, history_id, r"Results summary", occurrence=2)

        run1_rows = re.findall(r'^\s+([\d.E+\-]+)\s+([\d.E+\-]+)', run1_text, re.MULTILINE)
        run2_rows = re.findall(r'^\s+([\d.E+\-]+)\s+([\d.E+\-]+)', run2_text, re.MULTILINE)

        assert run1_rows, "No data rows in run 1 .res"
        assert run2_rows, "No data rows in run 2 .res"

        run1_final_temp = float(run1_rows[-1][1])
        run2_initial_temp = float(run2_rows[0][1])

        assert run2_initial_temp == pytest.approx(run1_final_temp, rel=1e-4), (
            f"Run 2 initial temp ({run2_initial_temp:.4f}K) does not match "
            f"Run 1 final temp ({run1_final_temp:.4f}K) — IC extraction may have failed"
        )

    def test_run2_ensi_has_ndttr(self, ic_chain_result):
        gi, history_id = ic_chain_result
        # Pattern anchored with $ to avoid matching "EnSight output … initial conditions (.ini)"
        raw = _download_dataset(gi, history_id, r"EnSight output \(\.ensi\.tar\.gz\)$", occurrence=2)
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        names = tf.getnames()
        tf.close()
        ndttr = [n for n in names if "NDTTR" in n]
        assert len(ndttr) > 0, f"No NDTTR files in run 2 output: {names}"


BC_SCHEDULE_WORKFLOW_TIMEOUT = 600


FIXTURE_MESH = "tests/fixtures/small_2x2x2.d"


@pytest.fixture(scope="module")
def bc_schedule_workflow_result(gi, bc_schedule_workflow_id):
    """Run the BC schedule workflow once and return (gi, history_id) for all tests."""
    history = gi.histories.create_history(name="test-bc-schedule")
    history_id = history["id"]

    invocation = gi.workflows.invoke_workflow(
        bc_schedule_workflow_id,
        inputs={},
        history_id=history_id,
        allow_tool_state_corrections=True,
    )
    invocation_id = invocation["id"]

    _wait_for_workflow(gi, invocation_id, history_id, BC_SCHEDULE_WORKFLOW_TIMEOUT, "BC schedule workflow")

    yield gi, history_id

    gi.histories.delete_history(history_id, purge=True)


class TestBCScheduleWorkflow:

    def test_workflow_imported(self, gi, bc_schedule_workflow_id):
        wf = gi.workflows.show_workflow(bc_schedule_workflow_id)
        assert wf["name"] == "GDPS Transient with BC Schedule"

    def test_bcs_file_has_step_headers(self, bc_schedule_workflow_result):
        gi, history_id = bc_schedule_workflow_result
        text = _download_text(gi, history_id, r"BC schedule \(\.bcs\)")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        int_lines = [l for l in lines if re.match(r'^\d+$', l)]
        assert "50" in int_lines, f"Step header '50' not found in .bcs: {int_lines}"
        assert "100" in int_lines, f"Step header '100' not found in .bcs: {int_lines}"

    def test_bcs_node_rows_sorted(self, bc_schedule_workflow_result):
        gi, history_id = bc_schedule_workflow_result
        text = _download_text(gi, history_id, r"BC schedule \(\.bcs\)")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        current_block = []
        for line in lines:
            if re.match(r'^\d+$', line):
                if current_block:
                    assert current_block == sorted(current_block), \
                        f"Node IDs not ascending in block: {current_block}"
                current_block = []
            else:
                parts = line.split()
                if len(parts) == 3:
                    current_block.append(int(parts[0]))
        if current_block:
            assert current_block == sorted(current_block), \
                f"Node IDs not ascending in last block: {current_block}"

    def test_res_no_nan(self, bc_schedule_workflow_result):
        gi, history_id = bc_schedule_workflow_result
        text = _download_text(gi, history_id, r"Results summary")
        assert "NaN" not in text

    def test_res_has_timestep_rows(self, bc_schedule_workflow_result):
        gi, history_id = bc_schedule_workflow_result
        text = _download_text(gi, history_id, r"Results summary")
        rows = re.findall(r'^\s+([\d.E+\-]+)\s+([\d.E+\-]+)', text, re.MULTILINE)
        assert len(rows) == 11, f"Expected 11 time output rows (t=0 + nstep/npri), got {len(rows)}"

    def test_res_temperatures_in_range(self, bc_schedule_workflow_result):
        gi, history_id = bc_schedule_workflow_result
        text = _download_text(gi, history_id, r"Results summary")
        rows = re.findall(r'^\s+([\d.E+\-]+)\s+([\d.E+\-]+)', text, re.MULTILINE)
        temps = [float(r[1]) for r in rows]
        assert all(285 <= t <= 710 for t in temps), \
            f"Temperatures outside expected range [285, 710]: {temps}"

    def test_ensi_has_ndttr_files(self, bc_schedule_workflow_result):
        gi, history_id = bc_schedule_workflow_result
        raw = _download_dataset(gi, history_id, r"EnSight output")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        names = tf.getnames()
        tf.close()
        ndttr = [n for n in names if "NDTTR" in n]
        assert len(ndttr) == 11, f"Expected 11 NDTTR files (t=0 + nstep/npri), got {len(ndttr)}: {names}"

    def test_vtu_tarball_valid(self, bc_schedule_workflow_result):
        gi, history_id = bc_schedule_workflow_result
        raw = _download_dataset(gi, history_id, r"VTK time series")
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        names = tf.getnames()
        tf.close()
        assert any(n.endswith(".pvd") for n in names), f"No .pvd in VTK tarball: {names}"
        assert any(n.endswith(".vtu") for n in names), f"No .vtu in VTK tarball: {names}"


SENSOR_TO_BCS_TIMEOUT = 120

SENSOR_CSV_CONTENT = """elapsed_seconds,upstream_temp,downstream_temp
10.0,650.0,293.0
20.0,700.0,293.0
"""


@pytest.fixture(scope="module")
def sensor_to_bcs_result(gi):
    """Run the sensor_to_bcs tool and return (gi, history_id, output_dataset_id)."""
    import io as _io
    import tempfile

    history = gi.histories.create_history(name="test-sensor-to-bcs")
    history_id = history["id"]

    mesh_upload = gi.tools.upload_file(FIXTURE_MESH, history_id)
    mesh_dataset_id = mesh_upload["outputs"][0]["id"]

    deadline = time.time() + 60
    while time.time() < deadline:
        ds = gi.datasets.show_dataset(mesh_dataset_id)
        if ds["state"] == "ok":
            break
        if ds["state"] == "error":
            pytest.fail(f"Mesh upload failed")
        time.sleep(2)
    else:
        pytest.fail("Mesh upload timed out")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp.write(SENSOR_CSV_CONTENT)
        tmp_path = tmp.name

    csv_upload = gi.tools.upload_file(tmp_path, history_id, file_type="txt")
    csv_dataset_id = csv_upload["outputs"][0]["id"]

    deadline = time.time() + 60
    while time.time() < deadline:
        ds = gi.datasets.show_dataset(csv_dataset_id)
        if ds["state"] == "ok":
            break
        if ds["state"] == "error":
            pytest.fail("CSV upload failed")
        time.sleep(2)
    else:
        pytest.fail("CSV upload timed out")

    result = gi.tools.run_tool(
        history_id=history_id,
        tool_id="gdps_sensor_to_bcs",
        tool_inputs={
            "mesh_d": {"src": "hda", "id": mesh_dataset_id},
            "sensor_csv": {"src": "hda", "id": csv_dataset_id},
            "zone_template": '[{"col_name": "upstream_temp", "axis_min": 0.0, "axis_max": 0.1}, {"col_name": "downstream_temp", "axis_min": 0.9, "axis_max": 1.0}]',
            "dtim": "10.0",
            "bc_mode|mode": "zone",
            "bc_mode|bc_axis": "z",
        },
    )
    output_dataset_id = result["outputs"][0]["id"]

    deadline = time.time() + SENSOR_TO_BCS_TIMEOUT
    while time.time() < deadline:
        ds = gi.datasets.show_dataset(output_dataset_id)
        if ds["state"] == "ok":
            break
        if ds["state"] == "error":
            _dump_errors(gi, history_id)
            pytest.fail("sensor_to_bcs tool run failed")
        time.sleep(3)
    else:
        pytest.fail(f"sensor_to_bcs tool timed out after {SENSOR_TO_BCS_TIMEOUT}s")

    yield gi, history_id, output_dataset_id

    gi.histories.delete_history(history_id, purge=True)


class TestSensorToBCSTool:

    def test_bcs_has_two_step_headers(self, sensor_to_bcs_result):
        gi, history_id, dataset_id = sensor_to_bcs_result
        data = gi.datasets.download_dataset(dataset_id)
        text = data.decode("utf-8", errors="replace")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        int_lines = [l for l in lines if re.match(r'^\d+$', l)]
        assert int_lines == ["1", "2"], \
            f"Expected step headers ['1', '2'], got {int_lines}"

    def test_bcs_temperatures_correct(self, sensor_to_bcs_result):
        gi, history_id, dataset_id = sensor_to_bcs_result
        data = gi.datasets.download_dataset(dataset_id)
        text = data.decode("utf-8", errors="replace")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        data_lines = [l for l in lines if not re.match(r'^\d+$', l)]
        temps = set()
        for line in data_lines:
            parts = line.split()
            if len(parts) == 3:
                temps.add(float(parts[2]))
        assert 650.0 in temps, f"650.0K not found in temperatures: {temps}"
        assert 700.0 in temps, f"700.0K not found in temperatures: {temps}"
        assert 293.0 in temps, f"293.0K cold BC not found in temperatures: {temps}"

    def test_bcs_nodes_ascending(self, sensor_to_bcs_result):
        gi, history_id, dataset_id = sensor_to_bcs_result
        data = gi.datasets.download_dataset(dataset_id)
        text = data.decode("utf-8", errors="replace")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        current_block = []
        for line in lines:
            if re.match(r'^\d+$', line):
                if current_block:
                    assert current_block == sorted(current_block), \
                        f"Node IDs not ascending in block: {current_block}"
                current_block = []
            else:
                parts = line.split()
                if len(parts) == 3:
                    current_block.append(int(parts[0]))
        if current_block:
            assert current_block == sorted(current_block), \
                f"Node IDs not ascending in last block: {current_block}"

PERMEATION_TIMEOUT = 600


@pytest.fixture(scope="module")
def permeation_steady_result(gi, permeation_steady_workflow_id):
    """Run the steady permeation workflow once."""
    history = gi.histories.create_history(name="test-permeation-steady")
    history_id = history["id"]

    invocation = gi.workflows.invoke_workflow(
        permeation_steady_workflow_id,
        inputs={},
        history_id=history_id,
        allow_tool_state_corrections=True,
    )
    _wait_for_workflow(gi, invocation["id"], history_id, PERMEATION_TIMEOUT,
                       "Steady permeation workflow")

    yield gi, history_id

    gi.histories.delete_history(history_id, purge=True)


@pytest.fixture(scope="module")
def permeation_transient_result(gi, permeation_transient_workflow_id):
    """Run the transient permeation workflow once."""
    history = gi.histories.create_history(name="test-permeation-transient")
    history_id = history["id"]

    invocation = gi.workflows.invoke_workflow(
        permeation_transient_workflow_id,
        inputs={},
        history_id=history_id,
        allow_tool_state_corrections=True,
    )
    _wait_for_workflow(gi, invocation["id"], history_id, PERMEATION_TIMEOUT,
                       "Transient permeation workflow")

    yield gi, history_id

    gi.histories.delete_history(history_id, purge=True)


class TestPermeationSteadyWorkflow:

    def test_workflow_imported(self, gi, permeation_steady_workflow_id):
        wf = gi.workflows.show_workflow(permeation_steady_workflow_id)
        assert wf["name"] == "Steady-State Hydrogen Permeation (Phase 5)"

    def test_workflow_step_count(self, gi, permeation_steady_workflow_id):
        wf = gi.workflows.show_workflow(permeation_steady_workflow_id)
        # 3 parameter inputs + 8 tool steps = 11 total
        assert len(wf["steps"]) == 11, \
            f"Expected 11 steps, got {len(wf['steps'])}"

    def test_diffusion_solver_converges(self, permeation_steady_result):
        gi, history_id = permeation_steady_result
        res = _download_text(gi, history_id, r"Results summary.*\.res")
        assert "NaN" not in res, "NaN in diffusion solver results"
        match = re.search(r"iterations to convergence was\s+(\d+)", res)
        assert match, f"No convergence info in .res:\n{res[:500]}"
        assert int(match.group(1)) < 200

    def test_concentration_field_produced(self, permeation_steady_result):
        gi, history_id = permeation_steady_result
        # occurrence=2: first is steady thermal (NDPTL), second is diffusion solver (NDTTR)
        raw = _download_dataset(gi, history_id, r"EnSight output \(\.ensi\.tar\.gz\)$", occurrence=2)
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        names = tf.getnames()
        tf.close()
        ndttr = [n for n in names if "NDTTR" in n]
        assert len(ndttr) > 0, f"No NDTTR (concentration) files in output: {names}"

    def test_no_nan_in_concentration(self, permeation_steady_result):
        gi, history_id = permeation_steady_result
        # occurrence=2: first is steady thermal (NDPTL), second is diffusion solver (NDTTR)
        raw = _download_dataset(gi, history_id, r"EnSight output \(\.ensi\.tar\.gz\)$", occurrence=2)
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        ndttr_members = sorted([m for m in tf.getmembers() if "NDTTR" in m.name], key=lambda m: m.name)
        content = tf.extractfile(ndttr_members[-1]).read().decode("utf-8", errors="replace")
        tf.close()
        vals = []
        for line in content.splitlines()[4:]:
            try:
                vals.append(float(line.strip()))
            except ValueError:
                pass
        assert vals, "No concentration values parsed from EnSight output"
        assert all(not (v != v) for v in vals), "NaN in concentration field"


class TestPermeationTransientWorkflow:

    def test_workflow_imported(self, gi, permeation_transient_workflow_id):
        wf = gi.workflows.show_workflow(permeation_transient_workflow_id)
        assert wf["name"] == "Transient Hydrogen Permeation (Phase 5)"

    def test_workflow_step_count(self, gi, permeation_transient_workflow_id):
        wf = gi.workflows.show_workflow(permeation_transient_workflow_id)
        # 2 parameter inputs + 8 tool steps = 10 total
        assert len(wf["steps"]) == 10, \
            f"Expected 10 steps, got {len(wf['steps'])}"

    def test_diffusion_solver_converges(self, permeation_transient_result):
        gi, history_id = permeation_transient_result
        res = _download_text(gi, history_id, r"Results summary.*\.res")
        assert "NaN" not in res, "NaN in transient diffusion solver results"

    def test_transient_concentration_field_produced(self, permeation_transient_result):
        gi, history_id = permeation_transient_result
        # occurrence=2: first match is steady thermal (p123/NDPTL), second is transient diffusion (p124/NDTTR)
        # Use $ anchor so "initial conditions (.ini)" datasets don't match
        raw = _download_dataset(gi, history_id, r"EnSight output \(\.ensi\.tar\.gz\)$", occurrence=2)
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz')
        names = tf.getnames()
        tf.close()
        ndttr = [n for n in names if "NDTTR" in n]
        assert len(ndttr) > 0, f"No NDTTR concentration files in transient output: {names}"
        assert len(ndttr) > 1, f"Expected multiple timesteps in transient output, got {len(ndttr)}"
