# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


# ================================
# Stage 1: grab MediaMTX binaries
# ================================
FROM bluenviron/mediamtx:1.15.6 AS mediamtx


# ================================
# Stage 2: runtime image
# ================================
FROM alpine:3.19

# ---- Install dependencies ----
RUN apk add --no-cache \
        ffmpeg \
        busybox-extras \
        curl

# ---- App directory ----
WORKDIR /opt/rtsp-streamer

# ---- Copy MediaMTX from stage 1 ----
COPY --from=mediamtx /mediamtx ./mediamtx
COPY --from=mediamtx /mediamtx.yml ./mediamtx.yml

# ---- Startup script ----
COPY start.sh ./start.sh
RUN chmod +x ./start.sh ./mediamtx

# ---- Runtime config ----
ENV MEDIA_DIR=/media \
    RTSP_PORT=8554 \
    VIDEO_URL="" \
    VIDEO_NAME="video.mp4" \
    STREAM_LOOP=true

EXPOSE 8554

# ---- Run ----
ENTRYPOINT ["/opt/rtsp-streamer/start.sh"]
