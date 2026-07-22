#!/usr/bin/env bash
###############################################################################
# Codex CLI / IDE extension setup via reverse proxy
#
# Usage:
#   bash scripts/utils/install_codex_proxy.sh \
#     --base-url https://your-reverse-proxy.example.com
#
#   bash scripts/utils/install_codex_proxy.sh \
#     --base-url https://your-reverse-proxy.example.com \
#     --proxy-script /data_4/liuyuan/lifebench/set_proxy.sh \
#     --auth-from /path/to/auth.json
#
#   bash scripts/utils/install_codex_proxy.sh --uninstall
#
# Notes:
# - This script keeps Codex on the official auth flow. It does not mint tokens.
# - If you already have a valid ChatGPT login at ~/.codex/auth.json, you usually
#   only need --base-url.
# - If the IDE extension is installed, this script points it at a wrapper binary
#   that injects proxy env and uses the bundled Codex CLI.
###############################################################################
set -euo pipefail

ACTION="install"
CUSTOM_CODEX_DIR=""
BASE_URL=""
HTTP_PROXY_URL=""
PROXY_SCRIPT="/data_4/liuyuan/lifebench/set_proxy.sh"
MODEL_NAME="gpt-5.5"
REASONING_EFFORT="high"
AUTH_FROM=""
CLI_WRAPPER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --uninstall)
            ACTION="uninstall"
            shift
            ;;
        --codex-dir)
            CUSTOM_CODEX_DIR="$2"
            shift 2
            ;;
        --codex-dir=*)
            CUSTOM_CODEX_DIR="${1#*=}"
            shift
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --base-url=*)
            BASE_URL="${1#*=}"
            shift
            ;;
        --http-proxy)
            HTTP_PROXY_URL="$2"
            shift 2
            ;;
        --http-proxy=*)
            HTTP_PROXY_URL="${1#*=}"
            shift
            ;;
        --proxy-script)
            PROXY_SCRIPT="$2"
            shift 2
            ;;
        --proxy-script=*)
            PROXY_SCRIPT="${1#*=}"
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model=*)
            MODEL_NAME="${1#*=}"
            shift
            ;;
        --reasoning-effort)
            REASONING_EFFORT="$2"
            shift 2
            ;;
        --reasoning-effort=*)
            REASONING_EFFORT="${1#*=}"
            shift
            ;;
        --auth-from)
            AUTH_FROM="$2"
            shift 2
            ;;
        --auth-from=*)
            AUTH_FROM="${1#*=}"
            shift
            ;;
        --cli-wrapper)
            CLI_WRAPPER="$2"
            shift 2
            ;;
        --cli-wrapper=*)
            CLI_WRAPPER="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--base-url URL] [--http-proxy URL] [--proxy-script PATH] [--codex-dir PATH] [--auth-from PATH] [--uninstall]"
            exit 1
            ;;
    esac
done

if [[ -n "$CUSTOM_CODEX_DIR" ]]; then
    CODEX_DIR="$CUSTOM_CODEX_DIR"
else
    CODEX_DIR="$HOME/.codex"
fi

if [[ -n "$CLI_WRAPPER" ]]; then
    WRAPPER_PATH="$CLI_WRAPPER"
else
    WRAPPER_PATH="$HOME/bin/codex"
fi

CONFIG_FILE="$CODEX_DIR/config.toml"
AUTH_FILE="$CODEX_DIR/auth.json"
PROXY_ENV_FILE="$CODEX_DIR/proxy_env.sh"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

find_extension_codex() {
    find "$HOME/.cursor-server/extensions" "$HOME/.vscode-server/extensions" \
        -maxdepth 4 -type f -path '*/openai.chatgpt-*/bin/linux-x86_64/codex' 2>/dev/null \
        | sort -V | tail -n 1
}

backup_file_if_exists() {
    local path="$1"
    if [[ -f "$path" ]]; then
        local backup="${path}.bak.$(date +%Y%m%d%H%M%S)"
        cp "$path" "$backup"
        info "  Backup created: $backup"
    fi
}

remove_generated_block() {
    local file="$1"
    local begin="$2"
    local end="$3"
    if [[ -f "$file" ]] && grep -q "$begin" "$file"; then
        python3 - "$file" "$begin" "$end" <<'PYEOF'
import sys
from pathlib import Path

path = Path(sys.argv[1])
begin = sys.argv[2]
end = sys.argv[3]
lines = path.read_text().splitlines()
out = []
skip = False
for line in lines:
    if line.strip() == begin:
        skip = True
        continue
    if skip and line.strip() == end:
        skip = False
        continue
    if not skip:
        out.append(line)
path.write_text("\n".join(out).rstrip() + "\n" if out else "")
PYEOF
    fi
}

if [[ "$ACTION" == "uninstall" ]]; then
    info "Uninstalling generated Codex reverse-proxy helper files..."
    rm -f "$PROXY_ENV_FILE"
    if [[ -f "$WRAPPER_PATH" ]] && grep -q "codex-proxy-wrapper" "$WRAPPER_PATH"; then
        rm -f "$WRAPPER_PATH"
        info "  Removed wrapper: $WRAPPER_PATH"
    fi
    remove_generated_block "$CONFIG_FILE" "# >>> codex-proxy-provider >>>" "# <<< codex-proxy-provider <<<"
    info "Done. Existing auth.json and non-generated config remain untouched."
    exit 0
fi

[[ -n "$BASE_URL" ]] || err "--base-url is required for install"

mkdir -p "$CODEX_DIR" "$(dirname "$WRAPPER_PATH")"

if [[ -n "$AUTH_FROM" ]]; then
    [[ -f "$AUTH_FROM" ]] || err "--auth-from file not found: $AUTH_FROM"
fi

if [[ -n "$PROXY_SCRIPT" ]]; then
    [[ -f "$PROXY_SCRIPT" ]] || err "--proxy-script file not found: $PROXY_SCRIPT"
fi

EXTENSION_CODEX="$(find_extension_codex)"
if [[ -z "$EXTENSION_CODEX" ]]; then
    warn "No bundled IDE Codex binary found under ~/.cursor-server or ~/.vscode-server"
    if command -v codex >/dev/null 2>&1; then
        EXTENSION_CODEX="$(command -v codex)"
        info "  Falling back to current codex in PATH: $EXTENSION_CODEX"
    else
        err "Cannot find a Codex binary. Install the Codex IDE extension or codex CLI first."
    fi
fi

info "Step 1/6: Preparing config directory..."
info "  Codex dir: $CODEX_DIR"
info "  Wrapper:   $WRAPPER_PATH"
info "  Base URL:  $BASE_URL"
if [[ -n "$PROXY_SCRIPT" ]]; then
    info "  Proxy script: $PROXY_SCRIPT"
fi

if [[ -n "$AUTH_FROM" ]]; then
    info "Step 2/6: Importing auth.json..."
    backup_file_if_exists "$AUTH_FILE"
    install -m 600 "$AUTH_FROM" "$AUTH_FILE"
    info "  Imported: $AUTH_FILE"
else
    info "Step 2/6: Keeping existing auth.json as-is"
    if [[ ! -f "$AUTH_FILE" ]]; then
        warn "  auth.json not found at $AUTH_FILE"
        warn "  You need a valid official ChatGPT/Codex auth.json or run codex login separately."
    fi
fi

info "Step 3/6: Writing Codex proxy config..."
backup_file_if_exists "$CONFIG_FILE"
python3 - "$CONFIG_FILE" "$MODEL_NAME" "$REASONING_EFFORT" "$BASE_URL" <<'PYEOF'
from pathlib import Path
import sys

config_file = Path(sys.argv[1])
model_name = sys.argv[2]
reasoning_effort = sys.argv[3]
base_url = sys.argv[4]
begin = "# >>> codex-proxy-provider >>>"
end = "# <<< codex-proxy-provider <<<"

existing = config_file.read_text() if config_file.exists() else ""

lines = existing.splitlines()
filtered = []
skip = False
for line in lines:
    if line.strip() == begin:
        skip = True
        continue
    if skip and line.strip() == end:
        skip = False
        continue
    if not skip:
        filtered.append(line)

def set_scalar(lines, key, value):
    prefix = f"{key} = "
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f'{key} = "{value}"'
            return
    lines.append(f'{key} = "{value}"')

set_scalar(filtered, "model", model_name)
set_scalar(filtered, "model_reasoning_effort", reasoning_effort)
set_scalar(filtered, "model_provider", "codex")

if filtered and filtered[-1] != "":
    filtered.append("")

filtered.extend([
    begin,
    '[model_providers.codex]',
    'name = "codex"',
    f'base_url = "{base_url}"',
    'wire_api = "responses"',
    'requires_openai_auth = true',
    end,
    "",
])

config_file.write_text("\n".join(filtered))
PYEOF
info "  Written: $CONFIG_FILE"

info "Step 4/6: Writing proxy environment helper..."
if [[ -n "$HTTP_PROXY_URL" ]]; then
    cat > "$PROXY_ENV_FILE" <<EOF
#!/usr/bin/env bash
# Generated by install_codex_proxy.sh
export HTTP_PROXY="$HTTP_PROXY_URL"
export HTTPS_PROXY="$HTTP_PROXY_URL"
export ALL_PROXY="$HTTP_PROXY_URL"
export http_proxy="$HTTP_PROXY_URL"
export https_proxy="$HTTP_PROXY_URL"
export all_proxy="$HTTP_PROXY_URL"
export CODEX_HOME="$CODEX_DIR"
EOF
    chmod 755 "$PROXY_ENV_FILE"
    info "  Written: $PROXY_ENV_FILE"
elif [[ -n "$PROXY_SCRIPT" ]]; then
    cat > "$PROXY_ENV_FILE" <<EOF
#!/usr/bin/env bash
# Generated by install_codex_proxy.sh
# shellcheck disable=SC1090
. "$PROXY_SCRIPT"
export CODEX_HOME="$CODEX_DIR"
EOF
    chmod 755 "$PROXY_ENV_FILE"
    info "  Written with sourced proxy script: $PROXY_ENV_FILE"
else
    cat > "$PROXY_ENV_FILE" <<EOF
#!/usr/bin/env bash
# Generated by install_codex_proxy.sh
export CODEX_HOME="$CODEX_DIR"
EOF
    chmod 755 "$PROXY_ENV_FILE"
    info "  Written minimal helper: $PROXY_ENV_FILE"
fi

info "Step 5/6: Writing Codex wrapper..."
cat > "$WRAPPER_PATH" <<EOF
#!/usr/bin/env bash
# codex-proxy-wrapper
set -euo pipefail

if [[ -f "$PROXY_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    . "$PROXY_ENV_FILE"
fi

exec "$EXTENSION_CODEX" "\$@"
EOF
chmod 755 "$WRAPPER_PATH"
info "  Wrapper target: $EXTENSION_CODEX"

info "Step 6/6: Updating IDE machine settings when present..."
find "$HOME" -maxdepth 3 \
    \( -path '*/.vscode-server/data/Machine/settings.json' -o -path '*/.cursor-server/data/Machine/settings.json' \) \
    -type f 2>/dev/null | while read -r settings_file; do
    info "  Updating $settings_file"
    python3 - "$settings_file" "$WRAPPER_PATH" "$HTTP_PROXY_URL" "$CODEX_DIR" <<'PYEOF'
import json
import os
import sys
from pathlib import Path

settings_path = Path(sys.argv[1])
wrapper_path = sys.argv[2]
http_proxy = sys.argv[3]
codex_dir = sys.argv[4]

if settings_path.exists():
    settings = json.loads(settings_path.read_text() or "{}")
else:
    settings = {}

settings["chatgpt.cliExecutable"] = wrapper_path
settings["chatgpt.openOnStartup"] = settings.get("chatgpt.openOnStartup", False)

term_env = settings.get("terminal.integrated.env.linux", {})
term_env["CODEX_HOME"] = codex_dir
if http_proxy:
    term_env["HTTP_PROXY"] = http_proxy
    term_env["HTTPS_PROXY"] = http_proxy
    term_env["ALL_PROXY"] = http_proxy
    term_env["http_proxy"] = http_proxy
    term_env["https_proxy"] = http_proxy
    term_env["all_proxy"] = http_proxy
settings["terminal.integrated.env.linux"] = term_env

if http_proxy:
    settings["http.proxy"] = http_proxy
    settings["http.proxySupport"] = "override"

settings.setdefault("terminal.integrated.profiles.linux", {})
settings["terminal.integrated.profiles.linux"]["bash (login)"] = {
    "path": "bash",
    "args": ["-l"],
}
settings["terminal.integrated.defaultProfile.linux"] = "bash (login)"

settings_path.parent.mkdir(parents=True, exist_ok=True)
settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
PYEOF
done

echo
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Codex reverse-proxy setup complete${NC}"
echo -e "${GREEN}================================================${NC}"
echo
echo "  Codex dir:   $CODEX_DIR"
echo "  Config:      $CONFIG_FILE"
echo "  Wrapper:     $WRAPPER_PATH"
echo "  Base URL:    $BASE_URL"
if [[ -n "$HTTP_PROXY_URL" ]]; then
    echo "  HTTP Proxy:  $HTTP_PROXY_URL"
fi
if [[ -n "$PROXY_SCRIPT" ]]; then
    echo "  Proxy script:$PROXY_SCRIPT"
fi
echo
echo "  Next steps:"
echo "    1. Ensure $AUTH_FILE contains your official ChatGPT/Codex login tokens."
echo "    2. Reload Cursor / VSCode window."
echo "    3. Run: $WRAPPER_PATH --version"
echo "    4. Optional: CODEX_HOME=\"$CODEX_DIR\" $WRAPPER_PATH login status"
