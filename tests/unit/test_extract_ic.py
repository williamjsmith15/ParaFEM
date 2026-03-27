import io
import os
import tarfile
import pytest
from galaxy.tools.gdps_extract_ic.gdps_extract_ic import extract_ic

def test_extract_ic_functional(tmp_path):
    # Create a dummy EnSight scalar file
    ensi_file = tmp_path / "test.ensi.NDTTR-000001"
    content = [
        "Description line",
        "part",
        "1",
        "coordinates",
        "1.0",
        "2.5",
        "3.0",
        "4.2"
    ]
    ensi_file.write_text("\n".join(content) + "\n")
    
    ini_file = tmp_path / "test.ini"
    
    extract_ic(str(ensi_file), str(ini_file))
    
    assert ini_file.exists()
    lines = ini_file.read_text().splitlines()
    assert lines[0] == "4"
    assert float(lines[1]) == 1.0
    assert float(lines[2]) == 2.5
    assert float(lines[3]) == 3.0
    assert float(lines[4]) == 4.2

def test_extract_ic_from_tarball(tmp_path):
    """extract_ic should accept a tar.gz and pick the last NDTTR by step number."""
    def make_ndttr(step, values):
        lines = ["Description", "part", "1", "coordinates"] + [str(v) for v in values]
        return "\n".join(lines).encode()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for step, values in [("000001", [1.0, 2.0]), ("000002", [3.0, 4.0])]:
            data = make_ndttr(step, values)
            info = tarfile.TarInfo(name=f"job.ensi.NDTTR-{step}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    buf.seek(0)

    tarball = tmp_path / "job.ensi.tar.gz"
    tarball.write_bytes(buf.read())

    ini_file = tmp_path / "out.ini"
    extract_ic(str(tarball), str(ini_file))

    lines = ini_file.read_text().splitlines()
    assert lines[0] == "2"
    assert float(lines[1]) == pytest.approx(3.0)
    assert float(lines[2]) == pytest.approx(4.0)


def test_extract_ic_with_fixture(tmp_path):
    fixture_path = "tests/fixtures/small_2x2x2.ensi.NDPTL-000001"
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture {fixture_path} not found")
        
    ini_file = tmp_path / "small_2x2x2.ini"
    extract_ic(fixture_path, str(ini_file))
    
    assert ini_file.exists()
    lines = ini_file.read_text().splitlines()
    # small_2x2x2 has 27 nodes (3x3x3 grid)
    assert lines[0] == "27"
    for val in lines[1:]:
        float(val) # should be parseable
