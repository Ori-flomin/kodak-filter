# Kodak Film Filter — Collaborative Photo Album

A Flask web app that transforms photos with realistic analog film looks and lets groups share a live photo album, joinable by QR code.

---

## Features

### Film Stocks
Seven physics-based film simulations, each with distinct grain, color response, and tonal curve:

| Stock | Character |
|---|---|
| **Kodak Gold 200** | Warm, orange-biased shadows, consumer film feel |
| **Kodak Tri-X 400** | Punchy B&W, heavy grain, strong halation |
| **Vintage 70s** | Expired slide film — milky shadows, heavy orange cast |
| **Cinematic** | Teal shadows / orange highlights Hollywood grade |
| **Fuji Velvia 50** | Hyper-saturated, vivid greens, almost no grain |
| **Kodak Portra 400** | Neutral, creamy, open highlights, beautiful skin tones |
| **Ilford HP5** | Soft B&W, open shadows, medium grain |

Each stock runs a full pipeline: **shadow lift → H&D toe/shoulder curve → color crossover → halation → organic grain clustering → channel blur → vignette**.

### Optional Overlays
- **Light Leak** — warm corner glow from randomly chosen edge
- **Date Stamp** — orange timestamp in the corner
- **Film Border** — dark frame with sprocket holes

### Live Preview
Load a photo, click any film stock — the filtered result appears instantly before you commit to saving.

### Collaborative Album
- Create an album, share it via **QR code** or link
- Everyone joins by scanning — no account needed
- Photos appear in real time for all members (5-second polling)
- Each photo is processed with the chosen film stock before saving

### Face Search
- Click **Faces** on any photo to detect all people in it
- Tap a face thumbnail to filter the gallery to photos containing that person
- Click **Find me in album** to upload your own photo and find yourself across the whole album
- Uses **MediaPipe BlazeFace** for detection + **eye-alignment** to normalise head pose + **LBP histogram embeddings** for matching

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3 + Flask |
| Image processing | Pillow + NumPy |
| Face detection | MediaPipe BlazeFace (Tasks API) |
| Database | SQLite (WAL mode) |
| QR codes | `qrcode[pil]` |
| Frontend | Vanilla HTML/CSS/JS — no frameworks |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the face detection model
```bash
python - <<'EOF'
import urllib.request
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "blaze_face.tflite"
)
print("Downloaded blaze_face.tflite")
EOF
```

### 3. Run
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Project Structure

```
kodak-filter/
├── app.py              # Flask routes + entire filter pipeline + face detection
├── db.py               # SQLite wrapper (albums, photos, faces, tags)
├── requirements.txt
├── blaze_face.tflite   # MediaPipe model (download separately, not in git)
├── templates/
│   ├── home.html       # Landing page — create or join an album
│   ├── album.html      # Main collaborative album view
│   ├── index.html      # Solo filter tool (legacy)
│   └── gallery.html    # Solo saved photos (legacy)
└── static/
    └── albums/         # Saved photos per album (not in git)
```

---

## How the Film Pipeline Works

Each photo passes through five ordered stages:

1. **Shadow lift** — raises the black floor asymmetrically per channel, simulating film base fog
2. **H&D curve** — applies a toe (compressed shadows) and shoulder (soft highlight rolloff), matching the characteristic curve of real film
3. **Color grading** — per-stock channel multiplies, shadow/highlight crossover, selective desaturation
4. **Halation** — bright highlights bleed a warm glow through the emulsion (red >> green >> blue)
5. **Organic grain** — Gaussian noise blended with a spatially clustered (blurred) copy, weighted by a midtone mask so grain is invisible in deep shadows and specular highlights

---

## Face Recognition Notes

Detection uses **MediaPipe BlazeFace** (neural network, handles angles and groups well).

Matching uses **LBP (Local Binary Pattern) histograms** — a classic illumination-robust face descriptor. Before computing the embedding, each face crop is **aligned using the eye keypoints** (rotated so eyes are horizontal), which normalises head tilt and significantly improves cross-photo matching.

Embeddings are computed from the **original photo before the film filter is applied**, so grain and color shifts don't affect recognition.

Limitation: extreme yaw (full profile) is not handled well without a deep learning face recognition model.
