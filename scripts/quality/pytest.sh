#!/usr/bin/env bash

set -euo pipefail

quality_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/quality/internal/lib.sh
source "${quality_dir}/internal/lib.sh"

lib_require_venv
lib_activate_venv
pytest "$@"
