#!/bin/bash
# SwiftBar plugin — administer the Local Model Server stack from the macOS menu bar.
# Symlink into your SwiftBar plugin folder (the "10s" = refresh every 10s):
#   ln -s "$PWD/local-model-server.10s.sh" "$HOME/Documents/bar plugins/local-model-server.10s.sh"
#
# <swiftbar.title>Local Model Server</swiftbar.title>
# <swiftbar.desc>Start/stop/restart and monitor the local OpenAI-compatible model server (gateway + engine).</swiftbar.desc>
# <swiftbar.refreshOnOpen>true</swiftbar.refreshOnOpen>

CTL="/Users/gui/dev/parakeet-mlx_guiapi/local_model_server_ctl.sh"

GW=$(curl -s --max-time 2 http://127.0.0.1:8123/health 2>/dev/null)   # gateway
EN=$(curl -s --max-time 2 http://127.0.0.1:8124/health 2>/dev/null)   # engine

gw_up=0; [ -n "$GW" ] && gw_up=1
en_up=0; [ -n "$EN" ] && en_up=1

# Engine model id (from gateway's backend model list, falls back to engine probe).
MODEL=$(echo "$GW" | sed -n 's/.*"default_model":"\([^"]*\)".*/\1/p')
[ -z "$MODEL" ] && MODEL=$(curl -s --max-time 2 http://127.0.0.1:8124/v1/models 2>/dev/null | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)

# Menu-bar icon: green if both up, orange if partial, red if both down.
if [ "$gw_up" = 1 ] && [ "$en_up" = 1 ]; then
    echo " | sfimage=brain sfcolor=#34C759"
elif [ "$gw_up" = 1 ] || [ "$en_up" = 1 ]; then
    echo " | sfimage=brain sfcolor=#FF9F0A"
else
    echo " | sfimage=brain sfcolor=#FF3B30"
fi
echo "---"
echo "Local Model Server"
echo "---"

if [ "$en_up" = 1 ]; then
    echo "Engine: running  ·  ${MODEL:-?} | color=#34C759 sfimage=checkmark.circle.fill"
    echo "Restart engine | bash=$CTL param1=engine param2=restart terminal=false refresh=true"
    echo "Stop engine | bash=$CTL param1=engine param2=stop terminal=false refresh=true"
else
    echo "Engine: stopped (:8124) | color=#FF3B30 sfimage=xmark.circle.fill"
    echo "Start engine | bash=$CTL param1=engine param2=start terminal=false refresh=true"
fi
echo "---"
if [ "$gw_up" = 1 ]; then
    echo "Gateway: running (:8123) | color=#34C759 sfimage=checkmark.circle.fill"
    echo "Restart gateway | bash=$CTL param1=gateway param2=restart terminal=false refresh=true"
    echo "Stop gateway | bash=$CTL param1=gateway param2=stop terminal=false refresh=true"
else
    echo "Gateway: stopped (:8123) | color=#FF3B30 sfimage=xmark.circle.fill"
    echo "Start gateway | bash=$CTL param1=gateway param2=start terminal=false refresh=true"
fi
echo "---"
echo "Start all | bash=$CTL param1=all param2=start terminal=false refresh=true"
echo "Restart all | bash=$CTL param1=all param2=restart terminal=false refresh=true"
echo "Stop all | bash=$CTL param1=all param2=stop terminal=false refresh=true"
echo "---"
echo "Open API docs | href=http://127.0.0.1:8123/docs"
echo "Engine logs | bash=$CTL param1=engine param2=logs terminal=false"
echo "Gateway logs | bash=$CTL param1=gateway param2=logs terminal=false"
echo "Refresh | refresh=true"
