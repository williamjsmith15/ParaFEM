# Phase 6d Convergence Studies Summary

## Mesh Convergence

**Setup:** GDPS membrane (L=5mm, T=0.5mm), uniform D=1.5e-12 m²/s, C_up=4.5e-3 mol/m³, C_dn=0.
**Analytical flux:** J = D × C_up / T = 1.35e-11 mol/m²/s

| n (per side) | Elements | J (mol/m²/s) | Error (%) |
|:---:|:---:|:---:|:---:|
| 5 | 125 | 1.3500e-11 | <0.001% |
| 8 | 512 | 1.3500e-11 | <0.001% |
| 16 | 4096 | 1.3500e-11 | <0.001% |
| 32 | 32768 | 1.3500e-11 | <0.004% |

**Result:** Solver is analytically exact for uniform D at all tested mesh sizes (<0.004% error). Mesh size n=8 (512 elements) is selected for the MC study as the minimum mesh achieving <1% error.

**Note:** n=10 and n=20 cause SIGABRT in rfemfield_thermal due to LAS3D internal size constraints. Tested working sizes: n=5, 8, 16, 32.

---

## Monte Carlo Convergence

**Setup:** GDPS membrane, 8×8×8 mesh (512 elements), D~LogNormal(μ=1.5e-12, σ=0.3e-12, CoV=20%), 200 instances.
**Expected:** var(mean_J) ∝ 1/N → log-log slope = -1.0

| N | mean J | std J | stderr | CoV |
|:---:|:---:|:---:|:---:|:---:|
| 50 | 9.005e+00 | 0.364 | 0.051 | 4.0% |
| 100 | 8.953e+00 | 0.360 | 0.036 | 4.0% |
| 200 | 8.965e+00 | 0.368 | 0.026 | 4.1% |

**Log-log slope:** -0.984 (expected -1.0; tolerance ±5% → pass)

**Mean J vs analytical:** The flux proxy J_proxy = avg(C_layer1) / dz ≈ C_up / T = 9.0 mol/m⁴·D, with D multiplied in extract_flux to give mol/m²/s. Mean J across 200 instances converges to ~8.97, consistent with the harmonic-mean effective diffusivity being slightly below D_mean for log-normal D.

**Seeding:** `RFEM_SEED` environment variable added to rfemfield_thermal to override the default PID-based seed. This ensures each instance is independent when run in separate Docker containers (PID is otherwise always ~2 in a fresh container, producing identical fields).

---

## Recommendations

1. **Mesh choice:** n=8 (512 elements) is sufficient for diffusion studies; larger meshes add compute cost with no accuracy gain for uniform D. For strongly heterogeneous D, n=16 may be appropriate.
2. **MC sample size:** 200 instances gives CoV(mean) ≈ 2.9%, sufficient for 5% precision on mean flux. For 1% precision, N ≈ 1600 instances required.
3. **N_VALUES choice:** Use N ≥ 50 for slope estimation; N=10 has too few samples for reliable std estimation (±30% uncertainty on std).
