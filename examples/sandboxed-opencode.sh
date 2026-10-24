#!/usr/bin/env bash
# sandboxed-opencode.sh: Executes OpenCode inside a native macOS Seatbelt sandbox
# locked down to the current workspace and talking to the local KerasHub MLX server.

set -e

WORKSPACE="$(pwd)"
USER_HOME="$HOME"

# Seatbelt profile
PROFILE="(version 1)
(allow default)

;; 1. Network: Allow localhost ONLY (to talk to local KerasHub API server), block external WAN
(deny network-outbound)
(allow network-outbound (to ip \"localhost:*\"))
(allow network-outbound (literal \"/private/var/run/mDNSResponder\"))
(allow network-outbound (regex #\"^/dev/\"))

;; 2. Filesystem Writes: Allow ONLY the current repo directory, /tmp, terminal devices, and opencode state
(deny file-write*)
(allow file-write* (subpath \"$WORKSPACE\"))
(allow file-write* (subpath \"/tmp\"))
(allow file-write* (subpath \"/private/tmp\"))
(allow file-write* (subpath \"/private/var/folders\"))
(allow file-write* (subpath \"/dev\"))
(allow file-write* (subpath \"$USER_HOME/.local/share/opencode\"))
(allow file-write* (subpath \"$USER_HOME/.cache/opencode\"))
(allow file-write* (subpath \"$USER_HOME/.config/opencode\"))

;; 3. Sensitive Reads: Block SSH, AWS, GPG, and credentials
(deny file-read* (subpath \"$USER_HOME/.ssh\"))
(deny file-read* (subpath \"$USER_HOME/.aws\"))
(deny file-read* (subpath \"$USER_HOME/.gnupg\"))
(deny file-read* (subpath \"$USER_HOME/.credentials\"))
"

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"

echo "🔒 Starting OpenCode in macOS Seatbelt Sandbox:"
echo "   • Model Endpoint: http://localhost:8000/v1 (KerasHub Qwen on Apple Silicon GPU)"
echo "   • Workspace:      $WORKSPACE (writes confined to this directory)"
echo "   • Network:        Localhost only (external internet blocked)"
echo "   • Credentials:    ~/.ssh, ~/.aws, ~/.gnupg completely blocked"
echo ""

exec /usr/bin/sandbox-exec -p "$PROFILE" opencode --model qwen2.5-coder "$@"
