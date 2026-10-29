#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$project_dir/venv/bin/python" ]]; then
    exec "$project_dir/venv/bin/python" "$project_dir/lib/cleanup.py"
fi
exec python3 "$project_dir/lib/cleanup.py"
