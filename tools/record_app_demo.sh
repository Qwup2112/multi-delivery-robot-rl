#!/bin/bash
# Records the real app.py being driven with mouse and keyboard on a virtual display (Linux).
#   tools/record_app_demo.sh [out.mp4] [vi|en]      e.g. tools/record_app_demo.sh figures_en/app_demo.mp4 en
# Needs: Xvfb, xdotool, ffmpeg and a Python with tkinter (PYTHON=/path/to/python to choose it).
# The clicks use fixed screen coordinates of the app at 1920x1080. Takes about 3 minutes.
set -u
OUT=${1:-figures/app_demo.mp4}; LANGOPT=${2:-vi}; PY=${PYTHON:-python3}; LOG=${TMPDIR:-/tmp}/app_demo.log
OUT=$(realpath -m "$OUT")
export DISPLAY=:99
pkill -f "app[.]py --lang" ; pkill -x Xvfb; pkill -x ffmpeg; sleep 1
Xvfb :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & sleep 1
cd "$(dirname "$0")/.."
$PY app.py --lang $LANGOPT > "$LOG" 2>&1 &
for i in $(seq 30); do W=$(xdotool search --onlyvisible --name "." 2>/dev/null | head -1); [ -n "$W" ] && break; sleep 0.5; done
sleep 4
xdotool windowmove $W 0 0 windowsize $W 1920 1080; sleep 2
xdotool mousemove 1700 1000; xdotool click 1; sleep 2   # focus the window
ffmpeg -loglevel error -y -f x11grab -framerate 20 -video_size 1920x1080 -draw_mouse 1 -i :99 \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart "$OUT" &
FF=$!
sleep 3
clk() { xdotool mousemove --sync $1 $2; sleep 0.5; xdotool click 1; sleep ${3:-1}; }
pick() { clk $1 $2 1.5; clk $3 $4 2; }   # open a drop-down list, then click an item
# ---- Level 1: live training, slowed down so the learning is visible ----
pick 270 49 270 72           # map S1
pick 255 402 230 497         # 1,000 episodes per update
xdotool mousemove --sync 184 423; sleep 0.3; xdotool mousedown 1; xdotool mousemove --sync 200 423; sleep 0.2; xdotool mousemove --sync 222 423; xdotool mouseup 1; sleep 1   # slow down
clk 89 374 38                          # Train (24-32 s with the slow-down)
clk 330 688 9                          # Agent run (P): robot drives the route
# ---- Level 4: live training at full speed ----
pick 270 49 270 128          # map S4
pick 255 402 230 424         # auto (200 updates)
xdotool mousemove --sync 222 423; sleep 0.3; xdotool mousedown 1; xdotool mousemove --sync 170 423; xdotool mouseup 1; sleep 1   # no slow-down
clk 89 374 32                          # Train 6M episodes
clk 697 19 4                           # overlay: value V
clk 595 19 2                           # overlay: policy
clk 330 688 22                         # Agent run (P) through the whole S4 route
# ---- Manual control and state codes ----
clk 118 654 2                          # back to S
for k in d d d s; do xdotool key $k; sleep 0.7; done
sleep 1
clk 160 819 0.5; xdotool type --delay 150 17419; sleep 0.8; xdotool key Return; sleep 3   # jump to state 17419
for i in 1 2 3; do clk 126 688 1.5; done   # agent single steps
sleep 3
kill -INT $FF; wait $FF
pkill -f "app[.]py --lang"
pkill -x Xvfb
echo "Saved $OUT"
