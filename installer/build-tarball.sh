#!/bin/sh
# Build the single-binary CLI tarball.
#
# Produces ./dist/alloy-arm64-darwin.tar.gz with the layout:
#
#   alloy/
#   ├── python/            (python-build-standalone)
#   ├── lib/               (alloy_cli, alloy_torch, alloy_metal, deps)
#   └── version
#
# Run from the repo root. Requires curl, tar, and uv.
#
# No Apple signing/notarization — release authenticity is the ed25519 signature
# applied in CI (release.yml) and verified by install.sh.
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$REPO_ROOT/dist"
STAGING="$(mktemp -d -t alloy-build)"
WORK="$STAGING/alloy"
PYTHON_VERSION="3.12.7"
PBS_DATE="20241016"  # python-build-standalone release date
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_DATE}/cpython-${PYTHON_VERSION}+${PBS_DATE}-aarch64-apple-darwin-install_only.tar.gz"

note() { printf '[build] %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "build script requires macOS host"
[ "$(uname -m)" = "arm64" ] || die "build script requires arm64 host"

mkdir -p "$DIST" "$WORK"

# ---- 1. fetch + unpack python-build-standalone ------------------------------

PBS_TARBALL="$STAGING/python.tar.gz"
note "downloading python-build-standalone ${PYTHON_VERSION}"
curl -fsSL "$PBS_URL" -o "$PBS_TARBALL"
mkdir -p "$WORK/python"
tar -xzf "$PBS_TARBALL" -C "$WORK/python" --strip-components=1

# ---- 2. install alloy packages + deps into lib/ ----------------------------

mkdir -p "$WORK/lib"
PYBIN="$WORK/python/bin/python3.12"

command -v uv >/dev/null 2>&1 || die "uv is required (curl -fsSL https://astral.sh/uv/install.sh | sh)"

# Build the single `alloy-kit` wheel (the same artifact published to PyPI) and
# install `alloy-kit[serve]` from it, so the tarball ships exactly what
# `pip install 'alloy-kit[serve]'` would. Installing the workspace source
# packages together into one --target silently drops alloy/'s pure-Python
# files (only the CMake-built extension survives), so always go via the wheel.
note "building the alloy-kit wheel"
WHEELDIR="$STAGING/wheels"
MACOSX_DEPLOYMENT_TARGET=13.0 uv build --quiet --wheel "$REPO_ROOT/packaging" \
    --python "$PYBIN" --out-dir "$WHEELDIR"
# The dist name is `alloy-kit`, so the wheel normalizes to `alloy_kit-*.whl`.
ALLOY_WHEEL="$(ls "$WHEELDIR"/alloy_kit-*.whl 2>/dev/null | head -1)"
[ -n "$ALLOY_WHEEL" ] || die "alloy-kit wheel build produced no wheel"

note "installing alloy-kit[serve] + deps into lib/"
uv pip install --quiet --python "$PYBIN" --target "$WORK/lib" "${ALLOY_WHEEL}[serve]"

# ---- 3. drop version marker --------------------------------------------------

CLI_VERSION="$(awk -F'"' '/^__version__/ {print $2}' "$REPO_ROOT/packages/alloy-cli/src/alloy_cli/version.py")"
[ -n "$CLI_VERSION" ] || die "could not read alloy-cli version"
printf 'alloy-cli %s\n' "$CLI_VERSION" > "$WORK/version"

# ---- 4. tar + emit -----------------------------------------------------------

TARBALL="$DIST/alloy-arm64-darwin.tar.gz"
note "creating $TARBALL"
tar -czf "$TARBALL" -C "$STAGING" alloy

note "size: $(/usr/bin/du -sh "$TARBALL" | awk '{print $1}')"
note "tarball: $TARBALL"
rm -rf "$STAGING"
