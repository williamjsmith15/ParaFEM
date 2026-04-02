#!/bin/bash
# Nuclear reset — tears down Galaxy completely (containers + volumes + images)
# and redeploys from scratch. Use when the instance is in a bad state or you
# need a guaranteed clean slate.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "WARNING: This will destroy all Galaxy data (workflows, histories, datasets)."
read -p "Continue? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "--- Tearing down ---"
cd "$SCRIPT_DIR"

docker compose --profile tunnel down -v 2>/dev/null || docker compose down -v
rm -f config/job_conf.xml
echo "Containers and volumes removed."

echo ""
echo "--- Removing Galaxy image cache ---"
docker image rm parafem-galaxy 2>/dev/null && echo "Removed parafem-galaxy image." || echo "No cached image to remove."

echo ""
echo "--- Redeploying ---"
export DOCKER_GID=$(stat -c %g /var/run/docker.sock)
export REPO_ROOT

echo "REPO_ROOT: $REPO_ROOT"
echo "DOCKER_GID: $DOCKER_GID"

envsubst '${REPO_ROOT}' < "$SCRIPT_DIR/config/job_conf.xml.tmpl" > "$SCRIPT_DIR/config/job_conf.xml"
echo "Generated config/job_conf.xml"

COMPOSE_PROFILES=""
if [ -f ".secret_token" ]; then
    echo "Found .secret_token — starting with Cloudflare tunnel"
    COMPOSE_PROFILES="--profile tunnel"
else
    echo "No .secret_token found — skipping Cloudflare tunnel (local access only)"
fi

docker compose $COMPOSE_PROFILES up -d --build

echo ""
echo "Waiting for Galaxy to become healthy..."
TIMEOUT=300
ELAPSED=0
until docker inspect --format='{{.State.Health.Status}}' parafem-galaxy 2>/dev/null | grep -q healthy; do
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo ""
        echo "ERROR: Galaxy did not become healthy within ${TIMEOUT}s"
        docker compose logs galaxy 2>/dev/null | tail -30
        exit 1
    fi
    printf "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo ""
echo "Galaxy is ready at http://localhost:8180"
if [ -f ".secret_token" ]; then
    echo "Cloudflare tunnel active — check your Cloudflare dashboard for the public URL"
fi
echo "Admin: admin@example.org / admin123"
echo ""
echo "Bootstrap is running in the background..."
echo "Check progress: docker compose logs -f galaxy-bootstrap"
