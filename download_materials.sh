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
done < "$MANIFEST"
