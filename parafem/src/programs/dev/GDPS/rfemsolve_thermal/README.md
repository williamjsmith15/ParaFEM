# rfemsolve_thermal

3D steady-state diffusion / thermal FEM solver for the RFEM (Random Finite Element
Method) pipeline. Solves the Laplace equation with spatially varying per-element
material properties (diffusivity D or conductivity k) for one Monte Carlo instance.

## Relationship to p123

`rfemsolve_thermal` is based on `p123` (Smith, Griffiths & Margetts,
*Programming the Finite Element Method*, 5th ed., Chapter 12) with two additions:

1. **Instance-based file naming** — reads `<model>-<instance>.{d,dat,mat}` so that
   multiple Monte Carlo instances can run concurrently in separate working directories
   without overwriting each other.

2. **Per-element material properties** — reads a `.mat` file produced by
   `rfemfield_thermal` and uses `prop(1, etype_pp(iel))` as the diffusivity for each
   element, rather than a uniform global value. `kx = ky = kz = D_iel`.

## Usage

```
mpirun -np <N> rfemsolve_thermal <model_name> <instance-id>
```

`<model_name>` is the base name without extension.
`<instance-id>` matches the suffix used by `rfemfield_thermal` (e.g., `001`).

## Input files

| File | Contents |
|------|----------|
| `<model>-<instance>.d` | Mesh geometry (from rfemfield_thermal) |
| `<model>-<instance>.dat` | Solver control parameters (rfemsolve format, `np_types = nels`) |
| `<model>-<instance>.mat` | Per-element diffusivity D (from rfemfield_thermal) |
| `<model>.bnd` | Restrained nodes — shared across all instances (`nr=0` means unused) |
| `<model>.fix` | Dirichlet BCs — shared across all instances (from rfembc_thermal) |

The `.bnd` and `.fix` files always use the base model name (without instance suffix)
so they are shared across all MC instances from the same mesh/BC configuration.

### `.dat` file format (rfemsolve format)

```
<element>  <meshgen>  <partitioner>  <np_types>
<nels>  <nn>  <nr>  <nip>  <nod>  <loaded_freedoms>  <fixed_freedoms>
<tol>  <limit>  <mises>
```

All values are list-directed (free-format). `np_types` equals `nels`
(one material type per element). `nr = 0` when all BCs are via `.fix`.

## Output files

| File | Contents |
|------|----------|
| `<model>-<instance>.res` | Convergence summary: node/eq counts, iterations, timings |
| `<model>-<instance>.ensi.NDPTL-000001` | Per-node scalar field (concentration / temperature) |

The EnSight scalar file can be visualised directly in ParaView or packaged by
`gdps_rfemsolve_thermal` Galaxy tool into a tarball.

## Physics

Solves the steady-state scalar transport equation:

```
∇·(D(x) ∇φ) = 0
```

where `D(x)` is the spatially varying diffusivity (per-element constant from the
random field) and `φ` is the scalar unknown (gas concentration or temperature).

Dirichlet boundary conditions are enforced via the penalty method:
`K_jj += penalty` for fixed DOF `j`, `r_j = penalty × val_j`.

## Solver

Preconditioned Conjugate Gradient (PCG) with diagonal (Jacobi) preconditioner,
element-by-element assembly via `gather`/`scatter`. Convergence criterion:
`||xnew - x|| / ||xnew|| < tol`.
