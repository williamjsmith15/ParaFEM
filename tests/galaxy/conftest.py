"""Fixtures for Galaxy integration tests."""

import os
import time

import pytest

try:
    from bioblend.galaxy import GalaxyInstance
except ImportError:
    GalaxyInstance = None

GALAXY_URL = os.environ.get("GALAXY_URL", "http://localhost:8180")
GALAXY_API_KEY = os.environ.get("GALAXY_API_KEY", "testing-api-key")


def galaxy_is_healthy(url):
    try:
        import urllib.request
        resp = urllib.request.urlopen(f"{url}/api/version", timeout=5)
        return resp.status == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def galaxy_url():
    return GALAXY_URL


@pytest.fixture(scope="session")
def galaxy_api_key():
    return GALAXY_API_KEY


@pytest.fixture(scope="session")
def gi(galaxy_url, galaxy_api_key):
    if GalaxyInstance is None:
        pytest.skip("bioblend not installed (pip install bioblend)")

    if not galaxy_is_healthy(galaxy_url):
        pytest.skip(f"Galaxy not running at {galaxy_url}")

    # Bootstrap key has limited access; get the admin user's real API key
    admin_gi = GalaxyInstance(url=galaxy_url, key=galaxy_api_key)
    for user in admin_gi.users.get_users():
        if user["email"] == "admin@example.org":
            user_key = admin_gi.users.get_or_create_user_apikey(user["id"])
            return GalaxyInstance(url=galaxy_url, key=user_key)

    pytest.fail("Admin user not found — bootstrap may not have run")


@pytest.fixture(scope="session")
def workflow_id(gi):
    """Find the steady-thermal workflow, waiting for bootstrap if needed."""
    deadline = time.time() + 120
    while time.time() < deadline:
        workflows = gi.workflows.get_workflows(published=True)
        for wf in workflows:
            if "Steady-State Thermal" in wf["name"]:
                return wf["id"]
        time.sleep(5)

    pytest.fail("Steady-State Thermal workflow not found (bootstrap may have failed)")


@pytest.fixture(scope="session")
def transient_workflow_id(gi):
    """Find the transient-thermal workflow, waiting for bootstrap if needed."""
    deadline = time.time() + 120
    while time.time() < deadline:
        workflows = gi.workflows.get_workflows(published=True)
        for wf in workflows:
            if wf["name"] == "Transient Thermal (ParaFEM p124)":
                return wf["id"]
        time.sleep(5)

    pytest.fail("Transient Thermal workflow not found (bootstrap may have failed)")


@pytest.fixture(scope="session")
def ic_chain_workflow_id(gi):
    """Find the transient IC chain workflow, waiting for bootstrap if needed."""
    deadline = time.time() + 120
    while time.time() < deadline:
        workflows = gi.workflows.get_workflows(published=True)
        for wf in workflows:
            if "Transient IC Chain" in wf["name"]:
                return wf["id"]
        time.sleep(5)

    pytest.fail("Transient IC Chain workflow not found (bootstrap may have failed)")
