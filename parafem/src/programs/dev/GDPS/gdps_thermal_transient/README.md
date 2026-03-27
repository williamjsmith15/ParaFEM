# gdps_thermal_transient

3D transient heat conduction solver for the GDPS (Gas Driven Permeation System)
digital twin pipeline. Parallel PCG version using 8-node hexahedral elements and
implicit time integration (theta method).

## Relationship to p124

`gdps_thermal_transient` is derived from `p124` (Smith, Griffiths & Margetts,
*Programming the Finite Element Method*, 5th ed., Chapter 12). It reads identical
input files and implements the same FEM algorithm, with three additions:

1. **BC enforcement fix** — p124 has a latent bug where the PCG residual at
   Dirichlet nodes is zeroed before the solve, causing BC nodes to converge to
   zero rather than their prescribed value when `val0 ≠ 0`. This program removes
   that residual-override block and adds an explicit post-PCG enforcement step
   so BC nodes always hold their prescribed temperature.

2. **Non-zero initial conditions** — controlled by an optional `.ctrl` sidecar
   file (see below).

3. **Checkpoint/restart** — writes a binary `.chk` file at the end of every run
   for chaining sequential simulation segments.

## Usage

```
mpirun -np <N> gdps_thermal_transient <jobname>
```

`<jobname>` is the base name shared by all input files (no extension).

## Input files

All formats are identical to p124. Required files:

| File | Contents |
|------|----------|
| `<jobname>.dat` | Solver control parameters (see `.dat` format below) |
| `<jobname>.d` | Nodal coordinates and element connectivity |
| `<jobname>.bnd` | Restrained node list (boundary nodes, 1 DOF per node) |
| `<jobname>.fix` | Prescribed temperatures (node, sense=1, value) |
| `<jobname>.mat` | Material properties (kx, ky, kz, rho, cp per material) |

Optional:

| File | Contents |
|------|----------|
| `<jobname>.lds` | Applied nodal heat sources/sinks |
| `<jobname>.ctrl` | Initial condition mode (see below) |
| `<jobname>.chk` | Checkpoint from a previous run (required for ic_mode 3) |
| `<jobname>.bcs` | Time-varying boundary condition schedule (see below) |

### `.dat` format

```
element  nels  nn  nr  nip  nod
nstep  npri  nres  meshgen  partitioner  np_types
val0  dtim  nstep  npri
theta  tol  limit
```

| Parameter | Description |
|-----------|-------------|
| `element` | Element type: `hexahedron` |
| `nels` | Total number of elements |
| `nn` | Total number of nodes |
| `nr` | Number of restrained nodes |
| `nip` | Number of Gauss integration points (typically 8) |
| `nod` | Nodes per element (8 for linear hex) |
| `nstep` | Number of time steps |
| `npri` | Output every `npri` steps (EnSight files + `.res` rows) |
| `nres` | Global equation number of the monitor node (written to `.res`) |
| `meshgen` | 1 = ParaFEM native, 2 = Abaqus ordering |
| `partitioner` | 1 = naive (default), 2 = METIS |
| `np_types` | Number of material types |
| `val0` | Uniform initial temperature (used by ic_mode 1) |
| `dtim` | Time step size (seconds) |
| `theta` | Time integration parameter (0.5 = Crank-Nicolson, 1.0 = fully implicit) |
| `tol` | PCG convergence tolerance |
| `limit` | Maximum PCG iterations per time step |

## Initial condition control file (`.ctrl`)

By default (`ic_mode 1`) the solution is initialised to `val0` from the `.dat`
file. To override this, create a `<jobname>.ctrl` file:

```
<ic_mode>
<path_to_ic_file>     ! required for ic_mode 2 and 3 only
```

| `ic_mode` | Behaviour | Second line |
|-----------|-----------|-------------|
| `1` | Uniform initial temperature = `val0` (default, no `.ctrl` needed) | — |
| `2` | Per-node temperatures from a `.ini` file | Path to `.ini` file |
| `3` | Restart from a binary `.chk` file written by a previous run | Path to `.chk` file |

The `.ctrl` file is only read by process 0 and broadcast to all others. If no
`.ctrl` file exists the solver runs in mode 1.

### `.ini` file format

Plain text, produced by the `gdps_extract_ic` tool:

```
<nn>          ! total node count
<val_node_1>
<val_node_2>
...
<val_node_nn>
```

One floating-point temperature value per line, ordered by global node number.
The header node count must match `nn` in the `.dat` file.

## Output files

| File | Contents |
|------|----------|
| `<jobname>.res` | Text summary: header, monitor-node rows (`time  temperature  iters`), timing |
| `<jobname>.ensi.NDTTR-000000` | Initial state (written before the first time step) |
| `<jobname>.ensi.NDTTR-NNNNNN` | Per-node temperatures every `npri` steps (EnSight Gold scalar format) |
| `<jobname>.npp` | Node/step/process metadata used by post-processing tools |
| `<jobname>.chk` | Binary checkpoint of the final solution vector (all processes) |

The EnSight files are consumed directly by ParaView or packaged into a tarball by
the `gdps_parafem2vtu_transient` Galaxy tool.

### `.chk` file format

Unformatted stream binary, written by `write_x_pp`:

```
[integer]  nstep           ! total steps completed (used as j_chk on restart)
[string]   "*FINAL"        ! label record
[integer]  nstep           ! step index repeated
[real*8]   xnew_pp(i)...   ! per-process distributed solution vector
```

Reading is handled by `read_x_pp(argv, npes, numpe, j_chk, xnew_pp)`. On restart
with ic_mode 3, `j_chk` is used to offset the `.ensi.NDTTR` step counter and the
`.res` time column so output from chained runs forms a continuous sequence.

### `.bcs` file format

The optional `<jobname>.bcs` file allows for time-varying boundary conditions.
If present, the solver will update the prescribed temperatures at specified
time steps. Initial conditions (at step 0) are always read from the `.fix` file;
the `.bcs` file defines changes for `step > 0`.

The format is a multi-block text file, aligned with the convention used by the
`inp2pf.awk` preprocessor for multi-step Abaqus analyses.

- Each block begins with a single integer on its own line, representing the
  time step `j` at which the new BCs should be applied.
- This is followed by a set of rows, one for each fixed freedom, with the
  format `<node> <sense> <value>`.
- The rows within each block **must be sorted in ascending order of node ID**,
  exactly matching the order in the original `.fix` file. The solver reads the
  values positionally.

Example `.bcs` file with two updates for a mesh with 3 fixed freedoms:
```
50
1 1 600.0
2 1 600.0
3 1 293.0
100
1 1 700.0
2 1 600.0
3 1 293.0
```

At time step `j=50`, the temperatures for nodes 1, 2, and 3 will be updated.
At `j=100`, they will be updated again. For all other time steps, the previously
set values are retained.
