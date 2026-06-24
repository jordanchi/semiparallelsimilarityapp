#!/usr/bin/env bash
# Copy app source from the parent research folder into this deploy directory.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$(cd "$(dirname "$0")" && pwd)"

copy() {
  cp "$ROOT/$1" "$DEST/$1"
}

copy semiparallelapp.py
copy alignment_scorers.py
copy amr_utils.py
rm -rf "$DEST/amr_metrics"
cp -R "$ROOT/amr_metrics" "$DEST/amr_metrics"

SAMPLE="$ROOT/Test csvs/arendt_hucon_de-en_sents.csv"
if [[ -f "$SAMPLE" ]]; then
  cp "$SAMPLE" "$DEST/arendt_hucon_de-en_sents.csv"
fi

echo "Synced app files into $DEST"
