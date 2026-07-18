#!/usr/bin/env bash
# Download MediaPipe Tasks model files used by the backend.
# Files are stored under ./models/mediapipe and are gitignored.

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models/mediapipe}"
mkdir -p "${MODELS_DIR}"

# Pose Landmarker (lite) - fast, good accuracy
POSE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
POSE_OUT="${MODELS_DIR}/pose_landmarker.task"

# Image Segmenter - selfie_multiclass (6 classes):
#   0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others(accessories)
SEG_URL="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
SEG_OUT="${MODELS_DIR}/selfie_multiclass.tflite"

download() {
  local url="$1"
  local out="$2"
  if [ -f "${out}" ] && [ -s "${out}" ]; then
    echo "[skip] ${out} already exists"
    return 0
  fi
  echo "[get]  ${url} -> ${out}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "${out}.tmp" "${url}" \
      && mv "${out}.tmp" "${out}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${out}.tmp" "${url}" \
      && mv "${out}.tmp" "${out}"
  else
    echo "ERROR: need curl or wget" >&2
    exit 1
  fi
  echo "[ok]   ${out} ($(du -h "${out}" | cut -f1))"
}

download "${POSE_URL}" "${POSE_OUT}"
download "${SEG_URL}"  "${SEG_OUT}"

echo "All MediaPipe models downloaded to ${MODELS_DIR}"
