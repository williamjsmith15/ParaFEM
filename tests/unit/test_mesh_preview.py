"""Unit tests for the gdps_mesh_preview tool."""

import os
import sys
import xml.etree.ElementTree as ET
import pytest
from unittest.mock import MagicMock, patch

# Add tool dir to path for import
TOOL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'galaxy', 'tools', 'gdps_mesh_preview'
)
sys.path.append(TOOL_DIR)

# Mock parafem_common before importing the tool
sys.modules['parafem_common'] = MagicMock()
import gdps_mesh_preview

@pytest.fixture
def tool_xml():
    xml_path = os.path.join(TOOL_DIR, 'gdps_mesh_preview.xml')
    tree = ET.parse(xml_path)
    return tree.getroot()

class TestMeshPreviewXml:
    def test_xml_structure(self, tool_xml):
        assert tool_xml.attrib['id'] == 'gdps_mesh_preview'
        assert tool_xml.find('.//requirements/container').text.strip() == 'williamjsmith15/parafem-bcgen:latest'
        
        params = [p.attrib['name'] for p in tool_xml.findall('.//inputs/param')]
        assert 'mesh_d' in params
        assert 'nset_file' in params
        
        outputs = [o.attrib['name'] for o in tool_xml.findall('.//outputs/data')]
        assert 'output_vtu' in outputs

class TestMeshPreviewLogic:
    @patch('gdps_mesh_preview.parse_d_file')
    @patch('gdps_mesh_preview.parse_nset_file')
    @patch('gdps_mesh_preview.build_vtu_tree')
    @patch('gdps_mesh_preview.write_vtu')
    def test_main_logic(self, mock_write, mock_build, mock_parse_nset, mock_parse_d):
        # Setup mocks
        mock_parse_d.return_value = ({1: [0,0,0], 2: [1,1,1]}, {1: [1,2]}, 'hex', 8)
        mock_parse_nset.return_value = {'SET1': {1}, 'SET2': {2}}
        
        # Mock sys.argv
        with patch.object(sys, 'argv', ['gdps_mesh_preview.py', '--mesh_d', 'm.d', '--nset_file', 'm.nset', '--output', 'm.vtu']):
            gdps_mesh_preview.main()
            
        # Verify calls
        mock_parse_d.assert_called_with('m.d')
        mock_parse_nset.assert_called_with('m.nset')
        
        # Check point_data passed to build_vtu_tree
        args, kwargs = mock_build.call_args
        point_data = kwargs['point_data']
        assert 'SurfaceID' in point_data
        # Node IDs are 1 and 2. Surface IDs should be 1.0 and 2.0 (based on sorted NSET names: SET1, SET2)
        assert point_data['SurfaceID'] == [1.0, 2.0]
        
        mock_write.assert_called_once()
