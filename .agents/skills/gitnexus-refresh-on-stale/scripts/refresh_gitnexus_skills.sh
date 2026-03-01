#!/usr/bin/env bash
set -euo pipefail

STATUS_OUTPUT="$(npx gitnexus status 2>&1 || true)"
printf '%s\n' "$STATUS_OUTPUT"

if printf '%s' "$STATUS_OUTPUT" | tr '[:upper:]' '[:lower:]' | grep -q 'stale'; then
  echo "[gitnexus-refresh] Index is stale. Running: npx gitnexus analyze"
  npx gitnexus analyze
else
  echo "[gitnexus-refresh] Index is not stale. Skipping analyze."
fi

SRC_DIR=".claude/skills/gitnexus"
DST_DIR=".agents/skills/gitnexus"

if [[ -d "$SRC_DIR" ]]; then
  echo "[gitnexus-refresh] Syncing generated skills: $SRC_DIR -> $DST_DIR"
  mkdir -p .agents/skills
  if [[ -e "$DST_DIR" ]]; then
    rm -rf "$DST_DIR"
  fi
  mv "$SRC_DIR" "$DST_DIR"
  echo "[gitnexus-refresh] Sync complete."
else
  echo "[gitnexus-refresh] No generated directory at $SRC_DIR; nothing to sync."
fi
