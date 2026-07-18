# Human Body Part Detection & Color Analysis

An AI-powered computer vision application that detects **12 human body parts** from images, videos, and live webcam streams, and identifies the **dominant color** on each detected region.

## 🎯 What It Does

This app uses **MediaPipe** (Google's computer vision framework) to:

1. **Detect a human** in your image/video/webcam feed
2. **Identify 12 body parts**: face, hair, neck, torso, left/right arm, left/right hand, left/right leg, left/right foot
3. **Analyze the dominant color** on each part (from 11 color categories: Red, Blue, Green, Yellow, Black, White, Gray, Brown, Purple, Orange, Pink)
4. **Draw bounding boxes** around each detected part with color labels
5. **Export results** as JSON with bounding box coordinates, confidence scores, RGB/HSV values, and color distribution percentages

### Example Output

For a person wearing a blue shirt and brown pants:
```
face:       Brown  (skin tone)
hair:       Brown  (dark hair)
torso:      Blue   (shirt)
left_arm:   Blue   (shirt sleeve)
right_leg:  Brown  (pants)
```

## ✨ Features

- **Image Analysis**: Upload any photo (JPG, PNG, WebP) → get annotated image + JSON
- **Video Analysis**: Upload a video (MP4, MOV, WebM) → get annotated video + per-part summary
- **Live Stream**: Real-time webcam processing with two modes:
  - **WebRTC mode**: True real-time (~15-30 FPS, requires localhost or HTTPS)
  - **Polling mode**: Works everywhere (~1 FPS, uses camera snapshots)
- **11 Color Categories**: Red, Blue, Green, Yellow, Black, White, Gray, Brown, Purple, Orange, Pink
- **Skin tone aware**: Fair/medium/dark skin all correctly classified as "Brown" (not "Red")
- **No backend server required**: Everything runs locally in a single Streamlit process

## 📋 Requirements

- **Python 3.10+** (3.11 recommended)
- **pip** (Python package manager)
- **Webcam** (for live streaming, optional)
- **~50 MB disk space** (for model weights)

**Supported platforms**: Windows, macOS, Linux (x86_64, ARM64)

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/human-body-segmentation.git
cd human-body-segmentation
```

### Step 2: Create a Virtual Environment

A virtual environment keeps dependencies isolated. Run:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs ~15 packages including MediaPipe, OpenCV, Streamlit, and PyTorch. Installation takes 2-5 minutes depending on your internet speed.

### Step 4: Download Model Weights

The app uses pre-trained MediaPipe models. Download them with:

```bash
bash scripts/download_models.sh
```

This downloads ~22 MB of model files to `models/mediapipe/`. If the script fails, you can manually download:
- Pose Landmarker: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
- Image Segmenter: https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite

Place both files in `models/mediapipe/`.

### Step 5: Run the App

```bash
streamlit run streamlit_app.py
```

Your browser should automatically open to http://localhost:8501. If it doesn't, open that URL manually.

**That's it!** The app is running. 🎉

##  How to Use

### Image Analysis

1. Click the **"🖼 Image"** tab
2. Click **"Choose an image"** and select a photo
3. Wait 1-3 seconds for processing
4. View the **annotated image** (with bounding boxes and labels)
5. See the **detected parts list** with dominant colors
6. Download the **result JSON** or **annotated image** using the buttons

**Tips**:
- Use clear, well-lit photos for best results
- Full-body or upper-body shots work best
- The person should be facing the camera

### Video Analysis

1. Click the **"🎬 Video"** tab
2. (Optional) Adjust the **"Process every Nth frame"** slider
   - `1` = analyze every frame (slow, detailed)
   - `5` = analyze every 5th frame (fast, summary)
3. Click **"Choose a video"** and select a video file
4. Click **"Analyze video"**
5. Watch the progress bar as frames are processed
6. View the **annotated video** and **per-part summary**
7. Download the **annotated video** or **result JSON**

**Tips**:
- Short videos (5-30 seconds) work best for testing
- Processing time: ~2-5 seconds per frame on a modern CPU
- Use the slider to speed up processing for long videos

### Live Stream

1. Click the **" Live Stream"** tab
2. Choose a capture mode:
   - **WebRTC (real-time)**: Best performance, requires modern browser
   - **Polling (works everywhere)**: Slower but compatible with all browsers
3. Click **"Start"** or **"Start capturing"**
4. Allow camera access when your browser asks
5. See the **annotated live feed** with real-time detections
6. View the **latest detections** list below the video

**Tips**:
- WebRTC mode works on `localhost` without HTTPS
- For remote access, you need HTTPS (use ngrok or similar)
- Polling mode captures a snapshot every time you click the camera button
- Good lighting improves detection accuracy

##  Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit_app'"

You're running an old version of the app. Make sure you're running:
```bash
streamlit run streamlit_app.py
```
Not:
```bash
streamlit run streamlit_app/app.py  # WRONG (old structure)
```

### "Port 8501 is not available"

Another Streamlit app is running. Kill it:
```bash
# On macOS/Linux
pkill -f streamlit

# On Windows
taskkill /F /IM streamlit.exe
```

Or run on a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### "Models not found" error

Run the download script:
```bash
bash scripts/download_models.sh
```

If it fails, check your internet connection and firewall. You can manually download the models (see Step 4 above).

### Webcam not working (Live Stream)

- Make sure no other app is using your webcam
- Try **Polling mode** instead of WebRTC
- Check browser permissions (camera access)
- Try a different browser (Chrome/Edge/Firefox recommended)

### Slow processing

- Image analysis: Normal (1-3 seconds per image)
- Video analysis: Use the "Process every Nth frame" slider (try 5 or 10)
- Live stream: Use WebRTC mode for real-time performance

### Colors look wrong

The app uses HSV color space for classification. Some edge cases:
- Very dark colors → classified as "Black"
- Very light/desaturated colors → classified as "White" or "Gray"
- Skin tones → always "Brown" (not "Red" or "Orange")

This is intentional to match common color naming.

##  Project Structure

```
human-body-segmentation/
├── streamlit_app.py          # Main app (single-file, runs everything)
├── app/                      # Core detection engine
│   ├── __init__.py
│   ├── config.py             # Model paths and settings
│   ├── core/                 # Detection logic
│   │   ├── body_parts.py     # 12-part definitions
│   │   ├── color_analyzer.py # K-means color extraction
│   │   ├── detector.py       # Orchestrator (combines all modules)
│   │   ├── pose.py           # MediaPipe Pose wrapper
│   │   ├── segmenter.py      # MediaPipe Image Segmenter wrapper
│   │   └── visualizer.py     # Bounding box drawing
│   └── utils/                # Utilities
│       ├── colors.py         # HSV → named color lookup
│       └── geometry.py       # Bounding box helpers
├── scripts/
│   └── download_models.sh    # Model downloader
├── models/
│   └── mediapipe/            # Model weights (gitignored)
│       ├── pose_landmarker.task
│       └── selfie_multiclass.tflite
├── images/                   # Sample images for testing
│   ├── sample1.jpg
│   └── sample2.jpg
── tests/                    # Unit tests
│   ├── test_colors.py
│   └── test_color_analyzer.py
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md                 # This file
```

## 🧪 Running Tests

```bash
# Activate venv first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run tests
python -m pytest tests/ -v
```

All 10 tests should pass.

## 📊 Technical Details

### How It Works

1. **Pose Estimation**: MediaPipe Pose Landmarker detects 33 body landmarks (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.)
2. **Body Part Segmentation**: MediaPipe Image Segmenter (`selfie_multiclass`) classifies each pixel into 6 categories: background, hair, body-skin, face-skin, clothes, others
3. **Bounding Box Computation**: For each of the 12 body parts, we compute a bounding box from the relevant landmarks
4. **Color Analysis**: We extract pixels within each part's bounding box, run K-means clustering (k=5) in HSV color space, and classify each cluster into one of 11 named colors
5. **Rendering**: We draw bounding boxes, color swatches, and labels on the original image

### Color Classification Rules

The classifier uses HSV thresholds:
- **Black**: V < 50
- **White**: V > 215 and S < 20
- **Gray**: S < 30 (desaturated)
- **Brown**: Warm hue (0-25) with S < 160 (catches skin tones)
- **Red/Orange/Yellow/Green/Blue/Purple/Pink**: Chromatic hues with S ≥ 160

### Performance

- **Image**: 1-3 seconds on a modern CPU
- **Video**: 2-5 seconds per frame (use frame skipping for long videos)
- **Live (WebRTC)**: 15-30 FPS on a modern CPU
- **Live (Polling)**: ~1 FPS

## 🤝 Contributing

Contributions welcome! Feel free to open issues or pull requests.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

##  Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for pose estimation and segmentation
- [Streamlit](https://streamlit.io/) for the web UI framework
- [OpenCV](https://opencv.org/) for image processing

## 📧 Support

For issues, please open a GitHub issue at: https://github.com/YOUR_USERNAME/human-body-segmentation/issues
