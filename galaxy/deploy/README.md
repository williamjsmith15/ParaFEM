# ParaFEM Galaxy Deploy

Runs a local Galaxy instance pre-loaded with the ParaFEM workflow tools.

## Prerequisites

- Docker (with socket access at `/var/run/docker.sock`)
- Docker Compose v2
- Python 3.8+ with `bioblend` installed (`pip install bioblend`) — for running tests only

The Galaxy container runs jobs by spawning sibling Docker containers via the host socket.
The `group_add: "998"` in `compose.yaml` grants the Galaxy process access to the socket —
check your system's `docker` group GID if this needs changing (`getent group docker`).

## Rebuild images

Images are hosted on Docker Hub under `williamjsmith15/`. Rebuild and push with:

```bash
cd /path/to/ParaFEM
bash galaxy/build-push.sh
```

This builds and pushes three images:
| Image | Purpose |
|---|---|
| `williamjsmith15/parafem:latest` | ParaFEM binaries (p12meshgen, p123, p124, pf2ensi, inp2pf) |
| `williamjsmith15/parafem-bcgen:latest` | BC generator tools (Python + numpy) |
| `williamjsmith15/parafem-vtu:latest` | VTU converter (Python) |

To build locally for testing without pushing (`:local` tags):

```bash
docker build -t parafem:local -f contianers/Dockerfile .
docker build -t parafem-bcgen:local -f contianers/Dockerfile.bcgen .
docker build -t parafem-vtu:local -f contianers/Dockerfile.vtu .
```

## Deploy

```bash
cd galaxy/deploy
docker compose up -d
```

Galaxy starts on `http://localhost:8180`. The bootstrap container runs automatically
on first start — it creates the admin user, imports both workflows, and publishes them.
Allow ~60 seconds for Galaxy to become healthy before the bootstrap runs.

Default credentials:
- URL: `http://localhost:8180`
- Email: `admin@example.org`
- Username: `admin`
- Password: `admin123`
- API key: `testing-api-key`

## Optional: Cloudflare tunnel

To expose the instance publicly via Cloudflare:

1. Create a `.secret_token` file in `galaxy/deploy/`:
   ```
   TUNNEL_TOKEN=your-cloudflare-tunnel-token
   ```
2. Start with the tunnel profile:
   ```bash
   docker compose --profile tunnel up -d
   ```

## Tear down

```bash
docker compose down          # stop containers, keep data volume
docker compose down -v       # stop and delete data volume (full reset)
```

## Run tests

Unit and integration tests run without Galaxy:

```bash
pytest tests/unit/ tests/integration/ -v
```

Galaxy end-to-end tests require the stack to be running:

```bash
# Galaxy must be up (docker compose up -d) before running these
pytest tests/galaxy/ -v
```

Full suite:

```bash
pytest tests/ -v
```

## Troubleshooting

**Bootstrap didn't run / workflows missing**
The bootstrap container exits after running once. To re-run it:
```bash
docker compose rm -f galaxy-bootstrap
docker compose up -d galaxy-bootstrap
```

**Jobs fail with "docker: permission denied"**
The Galaxy container needs access to the Docker socket. Check the GID of the `docker`
group on your host:
```bash
getent group docker
```
Update `group_add` in `compose.yaml` to match, then restart.

**Galaxy won't start / port conflict**
Port `8180` is mapped to Galaxy's internal `8080`. Change the host port in `compose.yaml`
if 8180 is already in use, and update `GALAXY_URL` when running tests:
```bash
GALAXY_URL=http://localhost:<port> pytest tests/galaxy/ -v
```
