import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def small_mesh_d():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2.d')


@pytest.fixture
def small_mesh_bnd():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2.bnd')


@pytest.fixture
def small_dat():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2.dat')


@pytest.fixture
def small_fix():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2.fix')


@pytest.fixture
def small_ensi_ndptl():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2.ensi.NDPTL-000001')


@pytest.fixture
def small_res():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2.res')


@pytest.fixture
def small_transient_dat():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2_transient.dat')


@pytest.fixture
def small_transient_mat():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2_transient.mat')


@pytest.fixture
def small_transient_long_dat():
    return os.path.join(FIXTURES_DIR, 'small_2x2x2_transient_long.dat')
