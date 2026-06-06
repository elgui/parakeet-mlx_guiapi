#!/bin/bash
# Lifecycle control for the Local Model Server stack.
#   gateway = FastAPI front door (com.gui.local-model-server,  :8123)
#   engine  = llama.cpp omni backend (com.gui.local-model-engine, :8124, gemma-4-qat)
#
#   ./local_model_server_ctl.sh <engine|gateway|all> <start|stop|restart|status|health|logs|install>
#
# Robust lifecycle via launchctl — never orphans a process. Used by the SwiftBar plugin
# (local-model-server.10s.sh) and usable directly from the CLI.

set -uo pipefail

DOMAIN="gui/$(id -u)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LA="$HOME/Library/LaunchAgents"

svc="${1:-all}"
cmd="${2:-status}"

label_for() { case "$1" in engine) echo "com.gui.local-model-engine";; gateway) echo "com.gui.local-model-server";; esac; }
port_for()  { case "$1" in engine) echo 8124;; gateway) echo 8123;; esac; }
log_for()   { case "$1" in engine) echo "$REPO_DIR/local_model_engine.log";; gateway) echo "$REPO_DIR/local_model_server.log";; esac; }

do_one() {
  local s="$1" action="$2"
  local label; label="$(label_for "$s")"
  local plist="$LA/$label.plist"
  case "$action" in
    start)   launchctl bootstrap "$DOMAIN" "$plist" ;;
    stop)    launchctl bootout "$DOMAIN" "$plist" ;;
    restart) launchctl kickstart -k "$DOMAIN/$label" 2>/dev/null || launchctl bootstrap "$DOMAIN" "$plist" ;;
    status)  launchctl print "$DOMAIN/$label" 2>/dev/null | grep -E "state =|pid =" | head -2 ;;
    health)  curl -s --max-time 3 "http://127.0.0.1:$(port_for "$s")/health" ; echo ;;
    logs)    /usr/bin/open -a Console "$(log_for "$s")" ;;
    install) cp "$REPO_DIR/$label.plist" "$plist" && chmod 600 "$plist" && echo "installed $plist" ;;
    *) echo "unknown action: $action" >&2; exit 2 ;;
  esac
}

case "$svc" in
  engine|gateway) do_one "$svc" "$cmd" ;;
  all)
    # Engine first so the gateway's backend is up before it serves.
    if [ "$cmd" = "stop" ]; then do_one gateway "$cmd"; do_one engine "$cmd"
    else do_one engine "$cmd"; do_one gateway "$cmd"; fi
    ;;
  *) echo "usage: $0 <engine|gateway|all> <start|stop|restart|status|health|logs|install>" >&2; exit 2 ;;
esac
