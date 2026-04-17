#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building wheel..."
cd "${SCRIPT_DIR}"
python -m build

echo "==> Building gpu_base image..."
docker build -t gpu_base -f Dockerfile.gpu-base .

echo "==> Building argus-lens server image..."
docker compose build "$@"

echo "==> Done. Run: docker compose up"
