cd /tools/genai-applications-sizing/common/rtsp-streamer
docker build -t rtsp-streamer-url .


docker run -d --name rtsp-streamer \
  -p 8554:8554 \
  -e VIDEO_URL="https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4" \
  rtsp-streamer-url


rtsp://localhost:8554/video
