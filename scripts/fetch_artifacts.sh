#!/usr/bin/env bash
# Fetch the artifact deposit and verify it against the manifest in this repository.
#
# The repository is the authority on what the deposit should contain: results/MD5SUMS is
# committed here, so a tampered or truncated download fails against a checksum you already
# have rather than against one the download supplied.
set -euo pipefail
cd "$(dirname "$0")/.."

DOI="${ARTIFACT_DOI:-}"
URL="${ARTIFACT_URL:-}"

if [ -z "$DOI" ] || [ -z "$URL" ]; then
  echo "The artifact deposit is not published yet; this script will name it when it is." >&2
  exit 2
fi

mkdir -p results
TARBALL=results.tar.zst

# A truncated download that reports success is the failure mode to design against: a
# zero-byte file left in place is 'already downloaded' forever, so delete before retrying.
for attempt in 1 2 3; do
  if [ -s "$TARBALL" ]; then break; fi
  rm -f "$TARBALL"
  echo "fetching (attempt $attempt): $URL"
  curl -fL --retry 3 --retry-delay 5 -o "$TARBALL" "$URL" || true
done
[ -s "$TARBALL" ] || { echo "ERROR: download failed or is empty." >&2; exit 1; }

tar --use-compress-program=unzstd -xf "$TARBALL"

echo "verifying against results/MD5SUMS ..."
md5sum -c results/MD5SUMS --quiet
echo "OK: every artifact matches the manifest."
