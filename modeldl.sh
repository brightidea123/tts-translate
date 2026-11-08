#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$project_dir/.venv/bin/python" "$project_dir/scripts/prepare_models.py" \
  --download --model whisper "$@"
