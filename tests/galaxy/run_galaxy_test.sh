#!/bin/bash
# Standalone script: start Galaxy, run workflow tests, tear down.
# Usage: ./tests/galaxy/run_galaxy_test.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/../../galaxy/deploy" && pwd)"

cleanup() {
    echo ""
    echo "Tearing down Galaxy..."
    cd "$DEPLOY_DIR" && ./stop.sh
}
trap cleanup EXIT

echo "=== Starting Galaxy ==="
cd "$DEPLOY_DIR"
./run.sh

echo ""
echo "=== Running Galaxy integration tests ==="
cd "$SCRIPT_DIR"

# Install test dependencies if needed
pip install --quiet bioblend pytest 2>/dev/null || true

pytest test_galaxy_workflow.py -v --tb=short
