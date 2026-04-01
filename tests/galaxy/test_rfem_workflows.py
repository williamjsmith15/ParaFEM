"""Galaxy E2E tests for RFEM MC workflows.

Requires Galaxy running at localhost:8180 with all RFEM tools loaded and
the RFEM MC workflows imported via bootstrap.
"""

import json
import time
import pytest


WORKFLOW_TIMEOUT = 600
POLL_INTERVAL   = 10


def _ds_state(ds):
    return ds.get("state") or ds.get("populated_state", "unknown")


def _wait_for_workflow(gi, invocation_id, history_id, timeout, label="Workflow"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        inv   = gi.invocations.show_invocation(invocation_id)
        state = inv["state"]
        if state == "scheduled":
            datasets = gi.histories.show_history(history_id, contents=True)
            if datasets:
                failed = [ds for ds in datasets if _ds_state(ds) == "error"]
                if failed:
                    for ds in failed:
                        try:
                            stderr = gi.datasets.show_dataset(ds["id"]).get("stderr", "")
                            print(f"  ERROR dataset '{ds['name']}': {stderr[:400]}")
                        except Exception:
                            pass
                    pytest.fail(f"{label}: {len(failed)} dataset(s) in error: "
                                f"{[d['name'] for d in failed]}")
                ok_states = ("ok", "deleted", "discarded")
                if all(_ds_state(ds) in ok_states for ds in datasets):
                    return
        elif state in ("cancelled", "failed"):
            pytest.fail(f"{label} invocation {state}")
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"{label} timed out after {timeout}s")


class TestRFEMToolsLoaded:

    RFEM_TOOLS = [
        "gdps_rfemcube",
        "gdps_rfembc_thermal",
        "gdps_rfemfield_thermal",
        "gdps_rfemsolve_thermal",
        "gdps_rfemmc_runner",
        "gdps_rfemmc_collect",
        "gdps_rfem_map",
    ]

    def test_rfem_tools_loaded(self, gi):
        tools    = gi.tools.get_tools()
        tool_ids = {t["id"] for t in tools}
        missing  = [t for t in self.RFEM_TOOLS if t not in tool_ids]
        assert not missing, f"Missing RFEM tools: {missing}"


class TestRFEMMCWorkflow:

    def test_workflow_imported(self, rfem_mc_workflow_id):
        assert rfem_mc_workflow_id is not None

    def test_workflow_has_five_steps(self, gi, rfem_mc_workflow_id):
        wf = gi.workflows.show_workflow(rfem_mc_workflow_id)
        assert len(wf["steps"]) == 5, \
            f"Expected 5 steps (cube,bc,runner,solver,collect), got {len(wf['steps'])}"

    def test_run_mc_workflow_small(self, gi, rfem_mc_workflow_id):
        """Run the MC workflow with default parameters to verify end-to-end."""
        hist = gi.histories.create_history(name="Test RFEM MC small")

        res = gi.workflows.invoke_workflow(
            rfem_mc_workflow_id,
            history_id=hist["id"],
        )
        _wait_for_workflow(gi, res["id"], hist["id"],
                           timeout=WORKFLOW_TIMEOUT, label="RFEM MC small")

        datasets = gi.histories.show_history(hist["id"], contents=True)

        csv_ds  = next((d for d in datasets if "MC Summary"    in d["name"] and d["state"] == "ok"), None)
        json_ds = next((d for d in datasets if "MC Statistics" in d["name"] and d["state"] == "ok"), None)
        png_ds  = next((d for d in datasets if "MC Histogram"  in d["name"] and d["state"] == "ok"), None)

        assert csv_ds  is not None, "MC Summary CSV not found in history"
        assert json_ds is not None, "MC Statistics JSON not found in history"
        assert png_ds  is not None, "MC Histogram PNG not found in history"

        csv_content  = gi.datasets.download_dataset(csv_ds["id"]).decode()
        json_content = json.loads(gi.datasets.download_dataset(json_ds["id"]).decode())

        assert "flux" in csv_content
        assert json_content["count"] >= 1
        assert json_content["mean"] > 0, \
            f"Mean flux should be positive, got {json_content['mean']}"


class TestRFEMMCComplexWorkflow:

    def test_complex_workflow_imported(self, rfem_mc_complex_workflow_id):
        assert rfem_mc_complex_workflow_id is not None

    def test_complex_workflow_has_seven_steps(self, gi, rfem_mc_complex_workflow_id):
        wf = gi.workflows.show_workflow(rfem_mc_complex_workflow_id)
        assert len(wf["steps"]) == 7, \
            f"Expected 7 steps, got {len(wf['steps'])}"
