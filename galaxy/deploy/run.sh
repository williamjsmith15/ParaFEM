#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export DOCKER_GID=$(stat -c %g /var/run/docker.sock)
export REPO_ROOT

echo "REPO_ROOT: $REPO_ROOT"
echo "DOCKER_GID: $DOCKER_GID"

# Generate job_conf.xml from template
envsubst '${REPO_ROOT}' < "$SCRIPT_DIR/config/job_conf.xml.tmpl" > "$SCRIPT_DIR/config/job_conf.xml"
echo "Generated config/job_conf.xml"

cd "$SCRIPT_DIR"
docker compose up -d --build

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
echo "Admin: admin@example.org / admin123"
echo ""
echo "Bootstrap is running in the background..."
echo "Check progress: docker compose logs -f galaxy-bootstrap"
