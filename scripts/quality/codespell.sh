#!/usr/bin/env bash

set -euo pipefail

quality_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/quality/internal/lib.sh
source "${quality_dir}/internal/lib.sh"

lib_require_venv
lib_activate_venv
lib_codespell_targets

if ! command -v codespell &>/dev/null; then
	pip install -q codespell
fi

codespell "${LIB_CODESPELL_TARGETS[@]}" \
	--skip="*.pth,*.safetensors,*.png,*.jpg,*.jpeg,*.git" \
	-L "nd,ans,crate,ser,te,fo,vulnerabilit"
