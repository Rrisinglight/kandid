#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
./.venv/bin/python build_analysis_pdf.py --notebook analysis.ipynb --figures figures --output analysis.pdf
echo "URL: http://$(hostname -I 2>/dev/null | awk '{print $1}')/analysis.pdf"
