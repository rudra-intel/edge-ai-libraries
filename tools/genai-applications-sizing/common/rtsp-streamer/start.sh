#!/bin/sh
set -eu

MEDIA_DIR=${MEDIA_DIR:-/media}
RTSP_PORT=${RTSP_PORT:-8554}
STREAM_LOOP=${STREAM_LOOP:-true}
VIDEO_URL=${VIDEO_URL:-""}
VIDEO_NAME=${VIDEO_NAME:-"video.mp4"}
MEDIAMTX_BIN=/opt/rtsp-streamer/mediamtx

# ---- Create media directory ----
mkdir -p "$MEDIA_DIR"

# ---- Download video from URL ----
if [ -n "$VIDEO_URL" ]; then
  dest="$MEDIA_DIR/$VIDEO_NAME"
  if [ ! -f "$dest" ]; then
    echo "Downloading video from $VIDEO_URL ..."
    curl -fSL --retry 3 --retry-delay 5 -o "$dest" "$VIDEO_URL"
    echo "Downloaded to $dest ($(du -h "$dest" | cut -f1))"
  else
    echo "Video already exists at $dest, skipping download"
  fi
else
  echo "VIDEO_URL is not set – expecting .mp4 files already present in $MEDIA_DIR"
fi

# ---- Validate video file ----
if [ ! -f "$MEDIA_DIR/$VIDEO_NAME" ]; then
  echo "Video file $MEDIA_DIR/$VIDEO_NAME not found" >&2
  echo "Set VIDEO_URL to a public video URL, or mount the video into $MEDIA_DIR" >&2
  exit 1
fi

# ---- Start MediaMTX RTSP server ----
"$MEDIAMTX_BIN" >/tmp/mediamtx.log 2>&1 &
mediamtx_pid=$!
pids="$mediamtx_pid"

# Wait for RTSP server to accept connections
retry=50
while ! nc -z 127.0.0.1 "$RTSP_PORT" >/dev/null 2>&1; do
  retry=$((retry - 1))
  if [ "$retry" -le 0 ]; then
    echo "RTSP server failed to start on port $RTSP_PORT" >&2
    cat /tmp/mediamtx.log >&2
    kill "$mediamtx_pid"
    wait "$mediamtx_pid" 2>/dev/null || true
    exit 1
  fi
  sleep 0.2
done
echo "MediaMTX RTSP server listening on port $RTSP_PORT"

# ---- Start ffmpeg stream ----
file="$MEDIA_DIR/$VIDEO_NAME"
stream_name=${VIDEO_NAME%.*}

loop_args=""
if [ "$STREAM_LOOP" = "true" ]; then
  loop_args="-stream_loop -1"
fi

echo "Starting RTSP stream /$stream_name from $file"
ffmpeg \
  -hide_banner \
  -loglevel info \
  -re \
  $loop_args \
  -i "$file" \
  -c copy \
  -rtsp_transport tcp \
  -f rtsp \
  "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
stream_pid=$!
pids="$pids $stream_pid"

# ---- Graceful shutdown ----
cleanup() {
  echo "Stopping RTSP streams"
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
  done
}

trap 'cleanup' INT TERM

status=0
for pid in $pids; do
  if ! wait "$pid"; then
    status=$?
  fi
done

exit $status
