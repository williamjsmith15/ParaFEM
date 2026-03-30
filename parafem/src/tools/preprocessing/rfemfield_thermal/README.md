# rfemfield_thermal

Preprocessing tool for the RFEM (Random Finite Element Method) pipeline.
Generates a spatially correlated log-normal random field of diffusivity (or thermal
conductivity) values and maps them onto a model mesh by element centroid proximity.
Produces per-element material files for one Monte Carlo instance.

## Relationship to rfemfield

`rfemfield_thermal` is derived from `rfemfield` but targets `nodof=1` scalar problems.
Key differences:

- `np_types = 1`: single scalar property per element (diffusivity D or conductivity k)
- `.rf` file does not require a Poisson ratio line (unlike the elastic version)
- Output `.mat` uses `kx` property name (single column, consumed by `rfemsolve_thermal`)
- The instanced `.dat` file sets `np_types = nels` (one material type per element)

## Usage

```
rfemfield_thermal <rfem_job_name> <model_job_name> <instance-id>
```

or, to generate the RF grid only (no model mapping):

```
rfemfield_thermal <rfem_job_name>
```

## Input files

| File | Contents |
|------|----------|
| `<rfem_job_name>.rf` | Random field control file (sim3de format) |
| `<model_job_name>.dat` | Model control file (rfemsolve format, from rfembc_thermal) |
| `<model_job_name>.d` | Model mesh geometry |

### `.rf` file format

```
sim3de
<output>          ! 1 = mat only, 2 = mat + RF mesh .d
<nels> <nxe> <nze>
<aa> <bb> <cc>    ! RF element sizes (m)
<thx> <thy> <thz> ! Correlation lengths (m)
<emn> <esd>       ! Log-normal mean and std-dev of D (m^2/s) or k (W/(m.K))
<varfnc>          ! Variogram function: dlavx3 | dlsep3 | dlspx3
```

Valid variogram functions (from gaf77 `sim3de`): `dlavx3` (locally averaged
exponential — recommended), `dlsep3` (separable exponential), `dlspx3` (separable power).

## Output files

| File | Contents |
|------|----------|
| `<model_job_name>-<instance-id>.d` | Mesh with per-element material IDs |
| `<model_job_name>-<instance-id>.dat` | Updated control file (`np_types = nels`) |
| `<model_job_name>-<instance-id>.mat` | Per-element D or k values |
| `<rfem_job_name>.res` | Statistics: arithmetic, geometric and harmonic mean of field |
| `<rfem_job_name>.mat` | RF grid material file (always written) |
| `<rfem_job_name>.d` | RF grid mesh (only when `output = 2`) |

## `.mat` file format

```
*MATERIAL <nels> 1
kx
<iel>  <D_value>
<iel>  <D_value>
...
```

One row per element, element ID and diffusivity value. Consumed by `rfemsolve_thermal`
via `read_materialValue`.

## Mapping method

Each model element is mapped to the RF grid element whose centre is nearest by
normalising the model element centroid into the unit cube and indexing into the
`nxe × nye × nze` RF grid. Elements at the boundary clamp to the nearest RF cell.

## Monte Carlo usage

To generate N independent instances, run `rfemfield_thermal` N times with different
`instance-id` values (e.g., `001`, `002`, ..., `100`). Each call uses a random seed
drawn from the system clock via `sim3de`. Pass each instanced archive to
`rfemsolve_thermal` for the per-instance solve.
