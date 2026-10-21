#!/usr/bin/env bash
# Install the pre-commit hook (run once per clone, by the repo owner).
set -eu
REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK="$(git rev-parse --git-path hooks)/pre-commit"
SOURCE="$REPO_ROOT/scripts/hooks/pre-commit"

if [ -e "$HOOK" ] && ! [ -L "$HOOK" -a "$(readlink "$HOOK")" = "$SOURCE" ]; then
  echo "A different pre-commit hook already exists at $HOOK; not overwriting it." >&2
  exit 1
fi
chmod +x "$SOURCE"
ln -sf "$SOURCE" "$HOOK"
echo "Installed: $HOOK -> $SOURCE"
