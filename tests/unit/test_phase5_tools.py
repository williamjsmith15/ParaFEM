import os
import sys
import json
import pytest
import math
from unittest.mock import MagicMock

# Add tools to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../galaxy/tools')))
from gdps_sievert_bc.gdps_sievert_bc import calculate_solubility, calculate_concentration
from gdps_compute_diffusivity.gdps_compute_diffusivity import calculate_diffusivity

def test_sievert_solubility():
    # Iron parameters
    S0 = 5.1e-3
    Es = 28600.0
    T = 500.0
    R = 8.314462618
    
    expected_S = S0 * math.exp(-Es / (R * T))
    assert calculate_solubility(S0, Es, T) == pytest.approx(expected_S)
    
    # Test zero temperature
    assert calculate_solubility(S0, Es, 0) == 0.0

def test_sievert_concentration():
    S0 = 5.1e-3
    Es = 28600.0
    T = 500.0
    P = 100000.0 # 1 bar
    
    S = calculate_solubility(S0, Es, T)
    expected_C = S * math.sqrt(P)
    
    assert calculate_concentration(S0, Es, T, P) == pytest.approx(expected_C)
    
    # Test negative pressure
    assert calculate_concentration(S0, Es, T, -100) == 0.0

def test_arrhenius_diffusivity():
    # Iron parameters
    D0 = 4.1e-8
    Ea = 4100.0
    T = 500.0
    R = 8.314462618
    
    expected_D = D0 * math.exp(-Ea / (R * T))
    assert calculate_diffusivity(D0, Ea, T) == pytest.approx(expected_D)
    
    # Test high temperature
    T_high = 1000.0
    expected_D_high = D0 * math.exp(-Ea / (R * T_high))
    assert calculate_diffusivity(D0, Ea, T_high) == pytest.approx(expected_D_high)
    assert expected_D_high > expected_D
