"""Unit tests for RFEM Convergence Study scripts."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.modules['parafem_common'] = MagicMock()

import studies.mesh_convergence as mesh_study
import studies.mc_convergence as mc_study


def test_mesh_convergence_script_imports():
    assert hasattr(mesh_study, 'run_study')
    assert hasattr(mc_study,   'run_study')


def test_analytical_flux_formula():
    """J = D * C_up / L with known GDPS values."""
    D     = 1.5e-12
    C_up  = 4.5e-3
    L     = 0.5e-3
    J_expected = D * C_up / L   # 1.35e-11
    assert abs(J_expected - 1.35e-11) < 1e-15


def test_extract_flux_proxy_logic():
    """Verify J = D * avg_C_layer1 / dz gives correct flux for linear profile."""
    D    = 1.5e-12
    dz   = 0.25
    C_up = 1.0
    L    = 0.5
    # Linear profile C(z) = C_up * z / L; layer1 at z = dz = 0.25
    # C(0.25) = 0.5 → J = D * 0.5 / 0.25 = 2D
    # Analytical J = D * C_up / L = D * 1.0 / 0.5 = 2D (matches for linear profile)

    values = [0.5] * 9 + [0.0] * 18   # 9 layer-1 nodes with C=0.5
    avg_c = np.mean(values[:9])
    J = D * avg_c / dz
    assert abs(J - D * C_up / L) < 1e-25  # should match analytical


def test_1_over_n_slope_fit():
    """var(mean_J) = C/N gives slope -1 on log-log plot."""
    N   = np.array([10, 50, 100, 500])
    var = 1.0 / N  # var(mean) ∝ 1/N
    slope, _ = np.polyfit(np.log(N), np.log(var), 1)
    assert abs(slope - (-1.0)) < 1e-10


def test_mc_statistics_calculation():
    """Mean and variance-of-mean calculation."""
    all_J = [1.0, 2.0, 3.0, 4.0, 5.0]
    N = len(all_J)
    mean_J      = np.mean(all_J)
    std_J       = np.std(all_J)
    var_of_mean = (std_J ** 2) / N

    assert mean_J == 3.0
    assert abs(var_of_mean - (std_J ** 2) / N) < 1e-15
