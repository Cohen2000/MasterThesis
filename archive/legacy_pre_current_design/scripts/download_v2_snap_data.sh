#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/raw"
cd "$ROOT/data/raw"
# Small/medium official SNAP additions that work directly with datasets.yaml.
wget -c https://snap.stanford.edu/data/sx-mathoverflow.txt.gz
wget -c https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
for f in sx-mathoverflow.txt.gz soc-sign-bitcoinotc.csv.gz; do gzip -t "$f"; done
printf '\nDownloaded optional v2 SNAP datasets into %s\n' "$PWD"
