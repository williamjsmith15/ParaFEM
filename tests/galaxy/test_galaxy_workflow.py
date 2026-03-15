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
    "gdps_meshgen_box",
    "gdps_mesh_import",
    "gdps_bc_thermal",
    "gdps_steady_thermal",
    "gdps_postprocess",
    "gdps_parafem2vtu",
]

WORKFLOW_TIMEOUT = 600  # 10 minutes max for full pipeline
POLL_INTERVAL = 10


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
        assert len(wf["steps"]) == 5


@pytest.fixture(scope="module")
def workflow_result(gi, workflow_id):
    """Run the workflow once and return (gi, history_id) for all tests."""
    history = gi.histories.create_history(name="test-steady-thermal")
    history_id = history["id"]

    invocation = gi.workflows.invoke_workflow(
        workflow_id,
        history_id=history_id,
    )
    invocation_id = invocation["id"]

    deadline = time.time() + WORKFLOW_TIMEOUT
    while time.time() < deadline:
        inv = gi.invocations.show_invocation(invocation_id)
        state = inv["state"]
        if state == "scheduled":
            datasets = gi.histories.show_history(history_id, contents=True)
            if datasets and all(
                ds["state"] in ("ok", "error", "deleted", "discarded")
                for ds in datasets
            ):
                failed = [ds for ds in datasets if ds["state"] == "error"]
                if failed:
                    _dump_errors(gi, history_id)
                    pytest.fail(
                        f"{len(failed)} dataset(s) in error state: "
                        f"{[d['name'] for d in failed]}"
                    )
                break
        elif state in ("cancelled", "failed"):
            _dump_errors(gi, history_id)
            pytest.fail(f"Workflow invocation {state}")
        time.sleep(POLL_INTERVAL)
    else:
        _dump_errors(gi, history_id)
        pytest.fail("Workflow timed out")

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


def _download_dataset(gi, history_id, name_pattern):
    datasets = gi.histories.show_history(history_id, contents=True)
    for ds in datasets:
        if re.search(name_pattern, ds["name"], re.IGNORECASE):
            return gi.datasets.download_dataset(ds["id"])
    available = [ds["name"] for ds in datasets]
    pytest.fail(f"No dataset matching '{name_pattern}'. Available: {available}")


def _download_text(gi, history_id, name_pattern):
    data = _download_dataset(gi, history_id, name_pattern)
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
