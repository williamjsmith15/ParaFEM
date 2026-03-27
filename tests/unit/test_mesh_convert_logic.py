"""Unit tests for the gdps_mesh_convert logic."""

import sys
import os
from unittest.mock import MagicMock, patch
import pytest

# Add tool dir to path so we can import the script
TOOL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'galaxy', 'tools', 'gdps_mesh_convert'
)
sys.path.append(TOOL_DIR)

# Mock meshio BEFORE importing the tool script which imports meshio
sys.modules['meshio'] = MagicMock()
import gdps_mesh_convert

class TestMeshConvertLogic:

    @patch('meshio.read')
    @patch('meshio.Mesh')
    def test_convert_mesh_filters_cells(self, mock_mesh_class, mock_read):
        # Setup mock mesh with some supported and some unsupported cells
        mock_input_mesh = MagicMock()
        mock_input_mesh.points = [1, 2, 3]
        
        supported_cell = MagicMock()
        supported_cell.type = "hexahedron"
        
        unsupported_cell = MagicMock()
        unsupported_cell.type = "line"
        
        mock_input_mesh.cells = [supported_cell, unsupported_cell]
        mock_read.return_value = mock_input_mesh
        
        # Call the conversion function
        gdps_mesh_convert.convert_mesh("input.msh", "output.inp")
        
        # Verify that meshio.Mesh was called with only the supported cell
        args, kwargs = mock_mesh_class.call_args
        cells_passed = kwargs['cells']
        assert len(cells_passed) == 1
        assert cells_passed[0].type == "hexahedron"

    @patch('meshio.read')
    def test_convert_mesh_fails_no_supported_cells(self, mock_read):
        # Setup mock mesh with NO supported cells
        mock_input_mesh = MagicMock()
        unsupported_cell = MagicMock()
        unsupported_cell.type = "line"
        mock_input_mesh.cells = [unsupported_cell]
        mock_read.return_value = mock_input_mesh
        
        # Should exit with code 1
        with pytest.raises(SystemExit) as e:
            gdps_mesh_convert.convert_mesh("input.msh", "output.inp")
        assert e.value.code == 1

    @patch('meshio.read')
    def test_convert_mesh_handles_exception(self, mock_read):
        # Force an exception during read
        mock_read.side_effect = Exception("Read error")
        
        with pytest.raises(SystemExit) as e:
            gdps_mesh_convert.convert_mesh("input.msh", "output.inp")
        assert e.value.code == 1
