"""Galaxy integration tests for the Mesh Pipeline (Phase 3).

Requires:
  - Galaxy running at localhost:8080 (or GALAXY_URL env var)
  - bioblend installed (pip install bioblend)

Run with: pytest tests/galaxy/test_mesh_pipeline.py -v --tb=short
"""

import os
import pytest

EXPECTED_MESH_TOOLS = [
    "gdps_mesh_convert",
    "gdps_mesh_import",
    "gdps_mesh_preview",
]

class TestMeshPipelineSetup:

    def test_tools_loaded(self, gi):
        tools = gi.tools.get_tools()
        tool_ids = {t["id"] for t in tools}
        missing = [t for t in EXPECTED_MESH_TOOLS if t not in tool_ids]
        assert not missing, f"Missing tools: {missing}"

    # Note: Full E2E workflow testing for Phase 3 requires a .ga file
    # for the mesh pipeline which hasn't been created yet.
    # Individual tool tests can be added here using gi.tools.run_tool.
