#!/bin/sh
# Alloy installer — drops the `alloy` CLI into /usr/local/.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/rayanht/alloy/main/installer/install.sh | sh
#   ALLOY_TARBALL=/path/to/local.tar.gz sh install.sh   # dev override
#   ALLOY_VERSION=v0.1.0 sh install.sh                   # pin version (tag)
set -eu

PREFIX="${PREFIX:-/usr/local}"
INSTALL_ROOT="$PREFIX/libexec/alloy"
BIN_DIR="$PREFIX/bin"
DEFAULT_MODEL="${ALLOY_DEFAULT_MODEL:-llama3.2:1b}"

# Release signing key (ed25519, SSH signature format). To rotate: regenerate the
# keypair, replace this line, update the ALLOY_SSH_SIGN_KEY repo secret.
ALLOY_SIGN_PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIES1QQYPu8Lh1YeDh6aPwHfhtvRm1LYXn6ak/CHiNPbO"

err() { printf '%s\n' "error: $*" >&2; exit 1; }
note() { printf '[alloy] %s\n' "$*"; }

# ---- preflight --------------------------------------------------------------

[ "$(uname -s)" = "Darwin" ] || err "Alloy ships for macOS only (got $(uname -s))."
[ "$(uname -m)" = "arm64" ] || err "Alloy requires Apple Silicon (arm64). Intel Macs are not supported."

case "$(sw_vers -productVersion 2>/dev/null)" in
    1[3-9].*|2[0-9].*|3[0-9].*) : ;;
    "") err "could not determine macOS version." ;;
    *) err "Alloy requires macOS 13 (Ventura) or newer (got $(sw_vers -productVersion))." ;;
esac

# Write-access preflight. On stock arm64 macOS, /usr/local requires sudo; on
# Apple-Silicon Homebrew systems users typically set PREFIX=/opt/homebrew.
mkdir -p "$BIN_DIR" 2>/dev/null || err "cannot create $BIN_DIR (try: sudo sh install.sh, or PREFIX=\$HOME/.local sh install.sh)."
[ -w "$BIN_DIR" ] || err "cannot write to $BIN_DIR (try: sudo sh install.sh, or PREFIX=\$HOME/.local sh install.sh)."
mkdir -p "$PREFIX/libexec" 2>/dev/null || err "cannot create $PREFIX/libexec (try: sudo sh install.sh, or PREFIX=\$HOME/.local sh install.sh)."
[ -w "$PREFIX/libexec" ] || err "cannot write to $PREFIX/libexec (try: sudo sh install.sh, or PREFIX=\$HOME/.local sh install.sh)."

# ---- fetch ------------------------------------------------------------------

if [ -n "${ALLOY_TARBALL:-}" ]; then
    note "using local tarball: $ALLOY_TARBALL"
    TARBALL="$ALLOY_TARBALL"
else
    VERSION="${ALLOY_VERSION:-latest}"
    REPO="${ALLOY_REPO:-rayanht/alloy}"
    TARBALL_NAME="alloy-arm64-darwin.tar.gz"
    # GitHub serves /releases/latest/download/... as a redirect to the most
    # recent published release; tag-pinned downloads go through
    # /releases/download/<tag>/...
    if [ "$VERSION" = "latest" ]; then
        URL="https://github.com/$REPO/releases/latest/download/$TARBALL_NAME"
    else
        URL="https://github.com/$REPO/releases/download/$VERSION/$TARBALL_NAME"
    fi
    TARBALL="$(mktemp -t alloy-tarball).tar.gz"
    note "downloading $URL"
    if ! curl -fsSL "$URL" -o "$TARBALL"; then
        err "download failed. Check your network or pin ALLOY_VERSION to an existing release tag."
    fi

    # Verify the detached ed25519 signature shipped beside the tarball, using the
    # pinned public key above and stock `ssh-keygen -Y`.
    SIG="$TARBALL.sig"
    note "fetching signature"
    curl -fsSL "$URL.sig" -o "$SIG" || err "signature download failed."
    if [ "${ALLOY_SKIP_VERIFY:-0}" = "1" ]; then
        note "WARNING: skipping signature verification (ALLOY_SKIP_VERIFY=1)."
    else
        command -v ssh-keygen >/dev/null 2>&1 || \
            err "ssh-keygen not found; cannot verify the release signature."
        ALLOWED="$(mktemp -t alloy-signers)"
        printf 'alloy-release %s\n' "$ALLOY_SIGN_PUBKEY" > "$ALLOWED"
        if ssh-keygen -Y verify -f "$ALLOWED" -I alloy-release -n file \
                -s "$SIG" < "$TARBALL" >/dev/null 2>&1; then
            note "signature verified"
        else
            rm -f "$ALLOWED"
            err "signature verification failed; refusing to install. The download may be corrupt or tampered. (Override at your own risk with ALLOY_SKIP_VERIFY=1.)"
        fi
        rm -f "$ALLOWED"
    fi
fi

[ -f "$TARBALL" ] || err "tarball not found: $TARBALL"

# ---- install ----------------------------------------------------------------

if [ -d "$INSTALL_ROOT" ]; then
    note "removing existing install at $INSTALL_ROOT"
    rm -rf "$INSTALL_ROOT"
fi

mkdir -p "$INSTALL_ROOT"
note "extracting to $INSTALL_ROOT"
tar -xzf "$TARBALL" -C "$INSTALL_ROOT" --strip-components=1

note "clearing quarantine xattr"
/usr/bin/xattr -dr com.apple.quarantine "$INSTALL_ROOT" 2>/dev/null || true

# ---- shell stubs ------------------------------------------------------------

mkdir -p "$BIN_DIR"
# ALLOY_HOME is resolved from the stub's real path at run-time. The quoted
# heredoc prevents expansion.
cat > "$BIN_DIR/alloy" <<'EOF'
#!/bin/sh
SELF="$0"
case "$SELF" in /*) : ;; *) SELF="$(pwd)/$SELF" ;; esac
SELF_DIR="$(cd "$(dirname "$SELF")" && pwd -P)"
ALLOY_HOME="$(cd "$SELF_DIR/../libexec/alloy" && pwd -P)"
PYTHONPATH="$ALLOY_HOME/lib:${PYTHONPATH:-}" \
exec "$ALLOY_HOME/python/bin/python3.12" -S -m alloy_cli "$@"
EOF
chmod +x "$BIN_DIR/alloy"

# ---- done -------------------------------------------------------------------

note "installed:"
note "  $BIN_DIR/alloy"
note "  $INSTALL_ROOT"
note ""
note "next: \`alloy serve -m $DEFAULT_MODEL\`, then point any OpenAI/Ollama client at http://127.0.0.1:11434"
