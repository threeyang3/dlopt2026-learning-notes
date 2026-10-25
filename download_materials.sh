#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$SCRIPT_DIR/source_manifest.tsv"

while IFS=$'\t' read -r lecture role original_path source_url target_path; do
  if [[ "$lecture" == "lecture" ]]; then
    continue
  fi

  destination="$SCRIPT_DIR/$target_path"
  mkdir -p "$(dirname "$destination")"

  if [[ -s "$destination" ]] && file "$destination" | grep -q 'PDF document'; then
    printf 'skip  %s\n' "$target_path"
    continue
  fi

  printf 'fetch %s\n' "$target_path"
  curl --location --fail --silent --show-error \
    --retry 3 --retry-delay 2 \
    --output "$destination" "$source_url"

  if ! file "$destination" | grep -q 'PDF document'; then
    printf 'error: downloaded file is not a valid PDF: %s\n' "$target_path" >&2
    exit 1
  fi
done < "$MANIFEST"

if [[ -f "$SCRIPT_DIR/checksums.sha256" ]]; then
  (
    cd "$SCRIPT_DIR"
    shasum -a 256 -c checksums.sha256
  )
fi
