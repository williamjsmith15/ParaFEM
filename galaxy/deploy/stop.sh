#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

docker compose down -v
rm -f config/job_conf.xml

echo "Galaxy stopped and volumes removed."
