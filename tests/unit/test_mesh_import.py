"""Unit tests for the gdps_mesh_import Galaxy tool XML definition."""

import os
import xml.etree.ElementTree as ET
import pytest

TOOL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'galaxy', 'tools', 'gdps_mesh_import'
)
TOOL_XML = os.path.join(TOOL_DIR, 'gdps_mesh_import.xml')


@pytest.fixture
def tool_root():
    tree = ET.parse(TOOL_XML)
    return tree.getroot()


class TestToolXmlStructure:

    def test_xml_parses(self, tool_root):
        assert tool_root.tag == 'tool'

    def test_tool_id(self, tool_root):
        assert tool_root.attrib['id'] == 'gdps_mesh_import'

    def test_has_docker_requirement(self, tool_root):
        container = tool_root.find('.//requirements/container')
        assert container is not None
        assert container.attrib['type'] == 'docker'
        assert container.text.strip() != ''

    def test_has_inp_file_input(self, tool_root):
        params = tool_root.findall('.//inputs/param')
        names = [p.attrib['name'] for p in params]
        assert 'inp_file' in names

    def test_inp_file_is_data_type(self, tool_root):
        params = tool_root.findall('.//inputs/param')
        inp_param = [p for p in params if p.attrib['name'] == 'inp_file'][0]
        assert inp_param.attrib['type'] == 'data'

    def test_has_renumber_select(self, tool_root):
        params = tool_root.findall('.//inputs/param')
        renumber = [p for p in params if p.attrib['name'] == 'renumber'][0]
        assert renumber.attrib['type'] == 'select'
        options = renumber.findall('option')
        values = [o.attrib['value'] for o in options]
        assert 'yes' in values
        assert 'no' in values

    def test_has_output_d(self, tool_root):
        outputs = tool_root.findall('.//outputs/data')
        names = [o.attrib['name'] for o in outputs]
        assert 'output_d' in names


class TestCommandTemplate:

    def test_command_contains_inp2pf(self, tool_root):
        command = tool_root.find('command').text
        assert 'inp2pf' in command

    def test_command_has_renumber_conditional(self, tool_root):
        command = tool_root.find('command').text
        assert '-renumber' in command
        assert '#if' in command

    def test_command_copies_input_and_moves_output(self, tool_root):
        command = tool_root.find('command').text
        assert 'cp' in command
        assert 'mv' in command
        assert 'job.inp' in command
        assert 'job.d' in command
