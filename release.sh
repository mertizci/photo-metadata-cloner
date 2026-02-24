#!/usr/bin/env bash
set -euo pipefail

# Release automation for noai-watermark.
# Default behavior bumps patch version by 1 and commits version files.

PYPROJECT="pyproject.toml"
INIT_PY="src/__init__.py"

show_help() {
  echo ""
  echo "Usage: ./release.sh [patch|minor|major|<version>]"
  echo ""
  echo "Default:"
  echo "  patch         (e.g. 0.1.19 -> 0.1.20)"
  echo ""
}

case "${1:-}" in
  -h|--help)
    show_help
    exit 0
    ;;
esac

require_tools() {
  command -v git >/dev/null 2>&1 || { echo "Error: git is not installed."; exit 1; }
  command -v sed >/dev/null 2>&1 || { echo "Error: sed is not installed."; exit 1; }
  command -v gh >/dev/null 2>&1 || { echo "Error: gh is not installed."; exit 1; }
}

check_paths() {
  [[ -f "$PYPROJECT" ]] || { echo "Error: $PYPROJECT not found."; exit 1; }
  [[ -f "$INIT_PY" ]] || { echo "Error: $INIT_PY not found."; exit 1; }
}

check_clean_git_state() {
  local has_issues=0

  if ! git diff --cached --quiet 2>/dev/null; then
    echo "  - staged changes present"
    has_issues=1
  fi

  if ! git diff --quiet 2>/dev/null; then
    echo "  - unstaged changes present"
    has_issues=1
  fi

  if [[ -n "$(git ls-files --others --exclude-standard 2>/dev/null)" ]]; then
    echo "  - untracked files present"
    has_issues=1
  fi

  if [[ "$has_issues" -ne 0 ]]; then
    echo "Error: working tree must be clean before release."
    exit 1
  fi
}

current_branch() {
  git rev-parse --abbrev-ref HEAD
}

current_version() {
  sed -n 's/^version = "\(.*\)"/\1/p' "$PYPROJECT" | head -n1
}

is_valid_semver() {
  [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

bump_version() {
  local cur="$1"
  local bump_type="$2"
  local major minor patch

  IFS='.' read -r major minor patch <<<"$cur"

  case "$bump_type" in
    major) echo "$((major + 1)).0.0" ;;
    minor) echo "${major}.$((minor + 1)).0" ;;
    patch) echo "${major}.${minor}.$((patch + 1))" ;;
    *) echo "Error: invalid bump type: $bump_type"; exit 1 ;;
  esac
}

target_version() {
  local old_ver="$1"
  local arg="${2:-patch}"

  case "$arg" in
    patch|minor|major)
      bump_version "$old_ver" "$arg"
      ;;
    *)
      if is_valid_semver "$arg"; then
        echo "$arg"
      else
        echo "Error: invalid version argument '$arg'. Use patch|minor|major|<x.y.z>."
        exit 1
      fi
      ;;
  esac
}

set_version() {
  local ver="$1"
  sed -i '' "s/^version = \".*\"/version = \"${ver}\"/" "$PYPROJECT"
  sed -i '' "s/^__version__ = \".*\"/__version__ = \"${ver}\"/" "$INIT_PY"
}

confirm_release() {
  local old_ver="$1"
  local new_ver="$2"
  local tag="$3"
  local branch="$4"

  echo ""
  echo "Release summary"
  echo "  branch : $branch"
  echo "  version: $old_ver -> $new_ver"
  echo "  tag    : $tag"
  echo ""
  printf "Continue? [y/N]: "

  local response
  read -r response
  case "$response" in
    [yY]|[yY][eE][sS]) ;;
    *)
      echo "Release cancelled."
      exit 0
      ;;
  esac
}

create_release() {
  local tag="$1"
  local version="$2"
  local branch="$3"

  git tag -a "$tag" -m "Release ${tag}"
  git push origin "$branch" --tags

  gh release create "$tag" \
    --title "v${version}" \
    --notes "Release ${version}" \
    --latest
}

main() {
  require_tools
  check_paths
  check_clean_git_state

  local branch old_ver new_ver tag commit_msg arg manual=false
  branch="$(current_branch)"
  old_ver="$(current_version)"
  arg="${1:-patch}"

  if ! is_valid_semver "$old_ver"; then
    echo "Error: current version '$old_ver' is not semantic version (x.y.z)."
    exit 1
  fi

  case "$arg" in
    patch|minor|major) ;;
    *) manual=true ;;
  esac

  new_ver="$(target_version "$old_ver" "$arg")"
  tag="v${new_ver}"

  if "$manual"; then
    if git tag -l "$tag" | grep -q "^${tag}$" || gh release view "$tag" >/dev/null 2>&1; then
      echo "Error: ${tag} already exists (tag or release). Choose a different version."
      exit 1
    fi
  else
    while git tag -l "$tag" | grep -q "^${tag}$" || gh release view "$tag" >/dev/null 2>&1; do
      echo "Warning: ${tag} already exists (tag or release), bumping to next patch..."
      new_ver="$(bump_version "$new_ver" "patch")"
      tag="v${new_ver}"
    done
  fi

  commit_msg="chore(release): bump version to ${new_ver}"

  confirm_release "$old_ver" "$new_ver" "$tag" "$branch"

  set_version "$new_ver"
  git add "$PYPROJECT" "$INIT_PY"
  git commit -m "$commit_msg"
  echo "Committed: $commit_msg"

  create_release "$tag" "$new_ver" "$branch"
  echo "Release completed: ${tag}"
}

main "${1:-}"
