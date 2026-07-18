"""Human Body Part Detection & Color Analysis — single-file Streamlit UI.

Run with:
    streamlit run streamlit_app.py

No FastAPI backend is required: this file loads the models directly and
does all inference locally.  Three modes are available via tabs:

  * Image   — upload an image, get annotated image + JSON.
  * Video   — upload a video, get annotated video + JSON.
  * Live    — webcam frames processed in real time.
"""
from __future__ import annotations

# --- 0.  Path bootstrap ---------------------------------------------------
# `streamlit run streamlit_app.py` is launched from the project root, but
# Streamlit's loader doesn't always put that directory on sys.path.  Adding
# it explicitly makes `import app.core.*` work no matter how it's started.
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- 1.  Imports ----------------------------------------------------------
import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# --- 2.  Page config ------------------------------------------------------
st.set_page_config(
    page_title="Human Body Part Detection",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- 3.  Model loading (cached, single instance) -------------------------
@st.cache_resource(show_spinner="Loading MediaPipe models (first run only)…")
def get_analyzer():
    from app.core.detector import HumanBodyAnalyzer

    return HumanBodyAnalyzer()


# --- 4.  Helpers ----------------------------------------------------------
def encode_png_b64(img_bgr: np.ndarray) -> str:
    import base64

    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_body_parts(parts: list[dict]) -> None:
    if not parts:
        st.warning("No body parts detected.")
        return
    cols = st.columns(2)
    for i, p in enumerate(parts):
        with cols[i % 2]:
            rgb = p["dominant_color"]["rgb"]
            sw = "#%02x%02x%02x" % tuple(rgb)
            st.markdown(
                f"""
                <div style="border:1px solid #ddd; border-radius:8px;
                            padding:10px; margin-bottom:8px;">
                  <div style="display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; width:18px; height:18px;
                                 background:{sw}; border:1px solid #333;
                                 border-radius:3px;"></span>
                    <b>{p['name']}</b>
                    <span style="color:#666;">conf {p['confidence']:.2f}</span>
                  </div>
                  <div style="font-size:0.85em; margin-top:4px;">
                    dominant: <b>{p['dominant_color']['name']}</b>
                    &nbsp;RGB {rgb}
                    &nbsp;HSV {p['dominant_color']['hsv']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_image(img_bgr: np.ndarray) -> dict:
    analyzer = get_analyzer()
    t0 = time.time()
    analysis = analyzer.analyze(img_bgr)
    annotated = analyzer.annotate(img_bgr, analysis)
    elapsed_ms = (time.time() - t0) * 1000

    parts_json: list[dict] = []
    for p in analysis.parts:
        parts_json.append({
            "name": p.part_name,
            "bbox": p.bbox.as_dict(),
            "confidence": round(p.confidence, 3),
            "dominant_color": {
                "name": p.color.dominant.name,
                "rgb": list(p.color.dominant.rgb),
                "hsv": list(p.color.dominant.hsv),
            },
            "color_distribution": [
                {
                    "name": c.name,
                    "rgb": list(c.rgb),
                    "percentage": round(c.percentage, 2),
                }
                for c in p.color.distribution
            ],
        })
    return {
        "image_info": {
            "width": int(img_bgr.shape[1]),
            "height": int(img_bgr.shape[0]),
            "channels": 3,
        },
        "person_detected": analysis.person_detected,
        "timestamp_ms": analysis.timestamp_ms,
        "elapsed_ms": round(elapsed_ms, 1),
        "body_parts": parts_json,
        "annotated_image_b64": encode_png_b64(annotated),
    }


# --- 5.  UI ---------------------------------------------------------------
st.title("🧍 Human Body Part Detection & Color Analysis")
st.caption(
    "AI-based detection of 12 human body parts with dominant-color "
    "extraction.  Models: MediaPipe Pose + Selfie Multiclass segmentation."
)

tab_image, tab_video, tab_live = st.tabs(
    ["🖼 Image", "🎬 Video", "📡 Live Stream"]
)

# ============================== IMAGE TAB ===============================
with tab_image:
    st.write(
        "Upload a clear full-body or upper-body photo.  The app detects "
        "12 body parts and reports the dominant color for each."
    )
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"],
        key="img_uploader",
    )
    if uploaded is not None:
        raw = uploaded.read()
        nparr = np.frombuffer(raw, dtype=np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                caption="Input",
                use_container_width=True,
            )

        with st.spinner("Detecting body parts…"):
            result = run_image(img_bgr)
        annotated_bgr = cv2.imdecode(
            np.frombuffer(
                __import__("base64").b64decode(result["annotated_image_b64"]),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

        with col2:
            st.image(
                cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Annotated — {result['elapsed_ms']:.0f} ms",
                use_container_width=True,
            )

        st.divider()
        st.subheader(f"Detections ({len(result['body_parts'])} parts)")
        render_body_parts(result["body_parts"])

        st.download_button(
            "📄 Download result JSON",
            data=json.dumps(result, indent=2),
            file_name="result.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "🖼 Download annotated image",
            data=__import__("base64").b64decode(result["annotated_image_b64"]),
            file_name="annotated.png",
            mime="image/png",
            use_container_width=True,
        )

        with st.expander("Full JSON", expanded=False):
            st.json(result, expanded=False)

# ============================== VIDEO TAB ===============================
with tab_video:
    st.write(
        "Upload a short video.  Every frame is analyzed; an annotated video "
        "and a per-part summary are returned."
    )
    sample_every = st.slider(
        "Process every Nth frame",
        1, 10, 1,
        help="Higher = faster, but coarser results.",
    )
    uploaded_vid = st.file_uploader(
        "Choose a video", type=["mp4", "mov", "webm", "avi", "mkv"],
        key="vid_uploader",
    )
    if uploaded_vid is not None and st.button("Analyze video", type="primary"):
        analyzer = get_analyzer()
        raw = uploaded_vid.read()
        tmp_in = Path("/tmp") / f"input_{int(time.time())}.bin"
        tmp_in.write_bytes(raw)

        cap = cv2.VideoCapture(str(tmp_in))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        tmp_out = Path("/tmp") / f"annotated_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_out), fourcc, fps, (w, h))

        progress = st.progress(0.0, text="Starting…")
        per_part: dict[str, dict] = {}
        frames_done = 0
        person_frames = 0
        last_analysis = None
        t0 = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frames_done % max(1, sample_every) == 0:
                    last_analysis = analyzer.analyze(frame)
                    if last_analysis.person_detected:
                        person_frames += 1
                    for p in last_analysis.parts:
                        slot = per_part.setdefault(p.part_name, {
                            "n": 0, "r": 0, "g": 0, "b": 0,
                            "h": 0, "s": 0, "v": 0, "conf": 0.0,
                            "frames": 0,
                        })
                        slot["n"] += 1
                        slot["frames"] += 1
                        r, g, b = p.color.dominant.rgb
                        hh, ss, vv = p.color.dominant.hsv
                        slot["r"] += r
                        slot["g"] += g
                        slot["b"] += b
                        slot["h"] += hh
                        slot["s"] += ss
                        slot["v"] += vv
                        slot["conf"] += p.confidence
                    progress.progress(
                        min(1.0, (frames_done + 1) / max(1, total)),
                        text=f"frame {frames_done + 1}/{total or '?'}",
                    )
                if last_analysis is not None:
                    frame_out = analyzer.annotate(frame, last_analysis)
                else:
                    frame_out = frame
                writer.write(frame_out)
                frames_done += 1
        finally:
            cap.release()
            writer.release()
        elapsed = time.time() - t0
        progress.progress(1.0, text=f"Done in {elapsed:.1f}s")

        st.success(
            f"Processed {frames_done} frames ({person_frames} with a "
            f"person) in {elapsed:.1f}s."
        )

        if per_part:
            st.subheader("Per-part summary")
            cols = st.columns(2)
            for i, (name, s) in enumerate(per_part.items()):
                with cols[i % 2]:
                    avg = [int(s["r"] / s["n"]), int(s["g"] / s["n"]), int(s["b"] / s["n"])]
                    sw = "#%02x%02x%02x" % tuple(avg)
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:8px;
                                    padding:10px; margin-bottom:8px;">
                          <div style="display:flex; align-items:center; gap:8px;">
                            <span style="display:inline-block; width:18px; height:18px;
                                         background:{sw}; border:1px solid #333;
                                         border-radius:3px;"></span>
                            <b>{name}</b>
                            <span style="color:#666;">conf {s['conf']/s['n']:.2f}
                                   · {s['frames']} frames</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with open(tmp_out, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button(
            "🎬 Download annotated video",
            data=video_bytes,
            file_name="annotated.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
        try:
            tmp_in.unlink()
            tmp_out.unlink()
        except Exception:
            pass

# ============================== LIVE TAB ================================
with tab_live:
    st.write(
        "Pick a mode and click **Start**.  Frames are processed by the "
        "loaded model directly inside this Streamlit process — no backend "
        "server required."
    )
    mode = st.radio(
        "Capture mode",
        ["WebRTC (real-time)", "Polling (works everywhere)"],
        horizontal=True,
        index=0,
        key="live_mode",
    )

    if mode.startswith("WebRTC"):
        try:
            import av
            from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
        except Exception as exc:  # noqa: BLE001
            st.error(f"streamlit-webrtc not available: {exc}")
            st.stop()

        st.info(
            "Click **Start** below and allow camera access.  The annotated "
            "feed will appear in the video element.  WebRTC needs "
            "localhost or HTTPS."
        )

        class LiveProcessor(VideoProcessorBase):
            def __init__(self) -> None:
                self._analyzer = get_analyzer()
                self._lock = threading.Lock()
                self.last_ms: Optional[float] = None
                self.last_parts: list[dict] = []

            def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                bgr = frame.to_ndarray(format="bgr24")
                t0 = time.time()
                with self._lock:
                    analysis = self._analyzer.analyze(bgr)
                    annotated = self._analyzer.annotate(bgr, analysis)
                self.last_ms = (time.time() - t0) * 1000.0
                self.last_parts = [
                    {
                        "name": p.part_name,
                        "confidence": round(p.confidence, 3),
                        "dominant_color": {
                            "name": p.color.dominant.name,
                            "rgb": list(p.color.dominant.rgb),
                        },
                    }
                    for p in analysis.parts
                ]
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        ctx = webrtc_streamer(
            key="hbs-live",
            video_processor_factory=LiveProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        if ctx.state.playing:
            proc: LiveProcessor = ctx.video_processor
            if proc and proc.last_parts:
                st.caption(
                    f"Last inference: {proc.last_ms:.0f} ms · "
                    f"{len(proc.last_parts)} parts"
                )
                render_body_parts(proc.last_parts)
        else:
            st.info("Click **Start** to begin streaming.")

    else:
        st.info(
            "Click the camera button to take a snapshot.  Each snapshot is "
            "analyzed and shown back annotated.  This mode is slow (~1 FPS) "
            "but works in any browser."
        )
        snap = st.camera_input("Snapshot", key="poll_snap")
        if snap is not None:
            raw = snap.read()
            nparr = np.frombuffer(raw, dtype=np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            with st.spinner("Analyzing snapshot…"):
                result = run_image(img_bgr)
            annotated_bgr = cv2.imdecode(
                np.frombuffer(
                    __import__("base64").b64decode(result["annotated_image_b64"]),
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )
            st.image(
                cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Annotated snapshot — {result['elapsed_ms']:.0f} ms",
                use_container_width=True,
            )
            st.subheader(f"Detections ({len(result['body_parts'])} parts)")
            render_body_parts(result["body_parts"])

# ============================== SIDEBAR =================================
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Models**
        - MediaPipe Pose Landmarker (33 landmarks)
        - MediaPipe Selfie Multiclass (6-class segmentation)

        **Body parts (12)**
        face, hair, neck, torso, left/right arm, left/right hand,
        left/right leg, left/right foot

        **Color categories (11)**
        Red, Blue, Green, Yellow, Black, White, Gray, Brown,
        Purple, Orange, Pink
        """
    )
    st.divider()
    st.caption("Single-file Streamlit app — no backend required.")
