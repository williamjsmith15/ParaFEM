"""Unit tests for the gdps_mesh_convert Galaxy tool XML definition."""

import os
import xml.etree.ElementTree as ET
import pytest

TOOL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'galaxy', 'tools', 'gdps_mesh_convert'
)
TOOL_XML = os.path.join(TOOL_DIR, 'gdps_mesh_convert.xml')


@pytest.fixture
def tool_root():
    tree = ET.parse(TOOL_XML)
    return tree.getroot()


class TestToolXmlStructure:

    def test_xml_parses(self, tool_root):
        assert tool_root.tag == 'tool'

    def test_tool_id(self, tool_root):
        assert tool_root.attrib['id'] == 'gdps_mesh_convert'

    def test_has_docker_requirement(self, tool_root):
        container = tool_root.find('.//requirements/container')
        assert container is not None
        assert container.attrib['type'] == 'docker'
        assert container.text.strip().startswith('williamjsmith15/parafem-meshio')

    def test_has_input_mesh_input(self, tool_root):
        params = tool_root.findall('.//inputs/param')
        names = [p.attrib['name'] for p in params]
        assert 'input_mesh' in names

    def test_input_mesh_is_data_type(self, tool_root):
        params = tool_root.findall('.//inputs/param')
        inp_param = [p for p in params if p.attrib['name'] == 'input_mesh'][0]
        assert inp_param.attrib['type'] == 'data'

    def test_has_input_format_param(self, tool_root):
        params = tool_root.findall('.//inputs/param')
        names = [p.attrib['name'] for p in params]
        assert 'input_format' in names

    def test_has_output_inp(self, tool_root):
        outputs = tool_root.findall('.//outputs/data')
        names = [o.attrib['name'] for o in outputs]
        assert 'output_inp' in names


class TestCommandTemplate:

    def test_command_calls_python_script(self, tool_root):
        command = tool_root.find('command').text
        assert 'python' in command
        assert 'gdps_mesh_convert.py' in command

    def test_command_has_input_output_params(self, tool_root):
        command = tool_root.find('command').text
        assert '--input' in command
        assert '--output' in command

    def test_command_has_format_conditional(self, tool_root):
        command = tool_root.find('command').text
        assert '--format' in command
        assert '#if' in command
