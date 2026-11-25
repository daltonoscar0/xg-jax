#!/usr/bin/env bash
# Regenerate requirements.lock from the active virtualenv.
#
# Run it from a venv that has `pip install -e ".[dev]"` done and nothing else,
# so the lock is the project's closure rather than whatever else was installed
# for an afternoon.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "lock.sh: activate the venv first (source .venv/bin/activate)" >&2
  exit 1
fi

{
  sed -n '/^#/p' requirements.lock
  pip freeze --exclude-editable | sort
} > requirements.lock.new

mv requirements.lock.new requirements.lock
echo "requirements.lock: $(grep -cv '^#' requirements.lock) pinned packages"
