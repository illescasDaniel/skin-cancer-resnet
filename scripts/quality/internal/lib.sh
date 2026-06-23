#!/usr/bin/env bash

# Shared helpers for scripts/quality/*.sh.

LIB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_REPO_ROOT="$(cd "${LIB_SCRIPT_DIR}/../../.." && pwd)"

lib_require_venv() {
	if [[ ! -d "${LIB_REPO_ROOT}/.venv" ]]; then
		echo "Missing .venv. Create it first: python3 -m venv .venv" >&2
		exit 1
	fi
}

lib_activate_venv() {
	cd "${LIB_REPO_ROOT}" || return
	# shellcheck disable=SC1091
	source ".venv/bin/activate"
}

lib_ruff_targets() {
	# shellcheck disable=SC2034  # consumed by callers after sourcing
	LIB_RUFF_TARGETS=(src scripts tests)
}

lib_shell_targets() {
	# shellcheck disable=SC2034  # consumed by callers after sourcing
	mapfile -t LIB_SHELL_TARGETS < <(
		find "${LIB_REPO_ROOT}" -name "*.sh" \
			-not -path "*/.venv/*" \
			| sort
	)
}

lib_codespell_targets() {
	# shellcheck disable=SC2034  # consumed by callers after sourcing
	LIB_CODESPELL_TARGETS=(README.md src scripts)
}

lib_require_shell_tools() {
	local missing=()
	command -v shellcheck >/dev/null 2>&1 || missing+=("shellcheck")
	command -v shfmt >/dev/null 2>&1 || missing+=("shfmt")
	if [[ "${#missing[@]}" -gt 0 ]]; then
		echo "Missing shell tools: ${missing[*]}" >&2
		echo "Install with: sudo apt-get install shellcheck && curl -sSL ... shfmt" >&2
		exit 1
	fi
}
