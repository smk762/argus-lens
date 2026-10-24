#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building wheel..."
cd "${SCRIPT_DIR}"
# uv >= 0.13.5 (pyproject required-version) so the build does not go through
# a pip that still trusts the compromised emSign intermediates.
uv build

echo "==> Building gpu_base image..."
docker build -t gpu_base -f Dockerfile.gpu-base .

echo "==> Building argus-lens server image..."
docker compose build "$@"

echo "==> Done. Run: docker compose up"
