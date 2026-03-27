# Mesh Import Test Fixture — Instructions

This file explains what needs to be created in Abaqus to produce the test fixture
for `TestMeshImport` integration tests.

## What we need

A small Abaqus `.inp` file representing a simple box mesh that we can use to verify
`gdps_mesh_import` (the `inp2pf` converter) works correctly end-to-end in Docker.

## Geometry spec

Create a **2x2x2 hex box** (matching the existing `small_2x2x2` fixtures):

| Property | Value |
|---|---|
| Geometry | 1m × 1m × 1m cube |
| Elements | 2 × 2 × 2 = 8 elements |
| Element type | **C3D8** (8-node linear hex) or **DC3D8** (thermal hex) — either works |
| Nodes | 27 nodes (3×3×3 grid) |
| Node IDs | Should be contiguous (inp2pf -renumber handles non-contiguous) |
| Part name | anything (e.g. `Part-1`) |
| Renumber | Yes — export with sequential IDs from 1, or leave to inp2pf |

## Export settings

In Abaqus CAE when writing the `.inp`:
- **Job → Write Input** or **File → Export → Model**
- Include: `*Node`, `*Element` sections
- No need for: loads, steps, boundary conditions, materials — geometry only is fine
- Format: standard ASCII `.inp`

## What the test will verify

Once saved as `tests/fixtures/small_2x2x2_import.inp`, the test will:

1. Run `inp2pf -renumber job.inp` inside the `williamjsmith15/parafem:latest` container
2. Check the output `job.d` file:
   - Contains `*NODES` section with **27 nodes**
   - Contains `*ELEMENTS` section with **8 elements**
   - Node coordinates span `[0, 1]` in X and Y, `[-1, 0]` in Z (ParaFEM convention)
   - Element connectivity has **8 nodes per row** (C3D8)
3. Feed the resulting `.d` file into the steady-state BC generator to confirm downstream compatibility

## Naming

Save the file as:

```
tests/fixtures/small_2x2x2_import.inp
```

## Note on coordinate convention

ParaFEM's Z-axis is **negative** — the mesh generator outputs Z ∈ [-dim_z, 0].
`inp2pf` handles the coordinate transform automatically, so export the geometry
in standard Abaqus coordinates (Z ∈ [0, 1]) and let the tool convert it.
