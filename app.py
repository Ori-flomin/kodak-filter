from flask import Flask, request, send_file, render_template, jsonify, abort
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from io import BytesIO
import os
import secrets
import string
from datetime import datetime
import qrcode
import db

_BASE        = os.path.dirname(__file__)
_DET_MODEL   = os.path.join(_BASE, 'det_500m.onnx')    # SCRFD face detector
_REC_MODEL   = os.path.join(_BASE, 'w600k_mbf.onnx')   # ArcFace MobileNet

_det_session = None
_rec_session = None
_EMB_BYTES   = 512 * 4    # 512-dim float32


def _get_sessions():
    global _det_session, _rec_session
    if _det_session is None:
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            _det_session = ort.InferenceSession(_DET_MODEL, sess_options=opts,
                                                providers=['CPUExecutionProvider'])
            _rec_session = ort.InferenceSession(_REC_MODEL, sess_options=opts,
                                                providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f'Face models not available: {e}')
    return _det_session, _rec_session


# ---------------------------------------------------------------------------
# SCRFD face detection helpers
# ---------------------------------------------------------------------------

def _scrfd_preprocess(pil_img, input_size=(640, 640)):
    """Resize + pad to input_size, return (blob, scale, (pad_x, pad_y))."""
    iw, ih   = pil_img.size
    scale    = min(input_size[0] / iw, input_size[1] / ih)
    new_w    = int(iw * scale)
    new_h    = int(ih * scale)
    resized  = pil_img.resize((new_w, new_h), Image.LANCZOS)
    canvas   = Image.new('RGB', input_size, (0, 0, 0))
    pad_x    = (input_size[0] - new_w) // 2
    pad_y    = (input_size[1] - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr      = np.array(canvas, dtype=np.float32)
    arr      = (arr - 127.5) / 128.0
    blob     = arr.transpose(2, 0, 1)[np.newaxis]   # NCHW
    return blob, scale, (pad_x, pad_y)


def _scrfd_postprocess(outputs, scale, pad, orig_size, conf_thresh=0.45, nms_thresh=0.4):
    """
    Decode SCRFD outputs → list of ((x1,y1,x2,y2), kps[5×2]) in original coords.
    Output layout: [score8, score16, score32, box8, box16, box32, kps8, kps16, kps32]
    """
    strides     = [8, 16, 32]
    num_anchors = 2
    input_size  = 640
    pad_x, pad_y = pad
    orig_w, orig_h = orig_size

    all_boxes, all_scores, all_kps = [], [], []

    for i, stride in enumerate(strides):
        fh = fw = input_size // stride

        scores  = outputs[i    ].reshape(-1)        # (fh*fw*na,)
        boxes   = outputs[i + 3].reshape(-1, 4)     # (fh*fw*na, 4)
        kpoints = outputs[i + 6].reshape(-1, 10)    # (fh*fw*na, 10)

        # Anchor centers: num_anchors anchors per grid cell
        cy_g, cx_g = np.mgrid[0:fh, 0:fw]
        cell_c  = np.stack([cx_g.ravel(), cy_g.ravel()], axis=1)   # (fh*fw, 2)
        centers = np.repeat(cell_c, num_anchors, axis=0).astype(np.float32)
        centers = (centers + 0.5) * stride                          # (n, 2)

        cx, cy = centers[:, 0], centers[:, 1]
        decoded = np.stack([
            cx - boxes[:, 0] * stride,
            cy - boxes[:, 1] * stride,
            cx + boxes[:, 2] * stride,
            cy + boxes[:, 3] * stride,
        ], axis=1)

        kps_dec = kpoints.reshape(-1, 5, 2).copy()
        kps_dec[:, :, 0] = kps_dec[:, :, 0] * stride + cx[:, np.newaxis]
        kps_dec[:, :, 1] = kps_dec[:, :, 1] * stride + cy[:, np.newaxis]

        mask = scores > conf_thresh
        all_boxes.append(decoded[mask])
        all_scores.append(scores[mask])
        all_kps.append(kps_dec[mask])

    if not any(len(b) for b in all_boxes):
        return []

    boxes  = np.vstack(all_boxes)
    scores = np.hstack(all_scores)
    kps    = np.vstack(all_kps)

    # NMS
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]; keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < nms_thresh]

    results = []
    for i in keep:
        b = boxes[i]
        x1o = int(np.clip((b[0] - pad_x) / scale, 0, orig_w))
        y1o = int(np.clip((b[1] - pad_y) / scale, 0, orig_h))
        x2o = int(np.clip((b[2] - pad_x) / scale, 0, orig_w))
        y2o = int(np.clip((b[3] - pad_y) / scale, 0, orig_h))
        k = kps[i].copy()
        k[:, 0] = (k[:, 0] - pad_x) / scale
        k[:, 1] = (k[:, 1] - pad_y) / scale
        results.append(((x1o, y1o, x2o, y2o), k))
    return results


def _detect_faces_in_img(pil_img):
    """Run SCRFD → return list of ((x1,y1,x2,y2), kps) in original image coords."""
    det_sess, _ = _get_sessions()
    if det_sess is None:
        return []
    iw, ih  = pil_img.size
    blob, scale, pad = _scrfd_preprocess(pil_img)
    outputs = det_sess.run(None, {det_sess.get_inputs()[0].name: blob})
    return _scrfd_postprocess(outputs, scale, pad, (iw, ih))


# ---------------------------------------------------------------------------
# ArcFace embedding
# ---------------------------------------------------------------------------

def _arcface_embed(crop_88):
    """
    Return 512-dim ArcFace embedding (bytes) for an 88×88 RGB PIL crop.
    ArcFace expects 112×112 BGR, mean-subtracted, CHW float32.
    """
    _, rec_sess = _get_sessions()
    if rec_sess is None:
        return None
    face = np.array(crop_88.resize((112, 112), Image.LANCZOS), dtype=np.float32)
    face = face[:, :, ::-1]            # RGB → BGR
    face = (face - 127.5) / 128.0
    blob = face.transpose(2, 0, 1)[np.newaxis]
    emb  = rec_sess.run(None, {rec_sess.get_inputs()[0].name: blob})[0][0]
    emb  = emb / (np.linalg.norm(emb) + 1e-6)
    return emb.astype(np.float32).tobytes()


def _face_crop(pil_img, bbox, kps, size=88):
    """Align face using eye keypoints, return size×size crop."""
    import math
    iw, ih    = pil_img.size
    le, re    = kps[0], kps[1]          # left eye, right eye keypoints
    angle     = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    eye_cx    = (le[0] + re[0]) / 2
    eye_cy    = (le[1] + re[1]) / 2
    rotated   = pil_img.rotate(-angle, resample=Image.BICUBIC,
                               center=(eye_cx, eye_cy), expand=False)
    x1, y1, x2, y2 = bbox
    pad = int(max(x2 - x1, y2 - y1) * 0.25)
    return rotated.crop((
        max(0, x1 - pad), max(0, y1 - pad),
        min(iw, x2 + pad), min(ih, y2 + pad),
    )).resize((size, size), Image.LANCZOS)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALBUM_DIR = os.path.join(os.path.dirname(__file__), 'static', 'album')
os.makedirs(ALBUM_DIR, exist_ok=True)

ALBUMS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'albums')
os.makedirs(ALBUMS_DIR, exist_ok=True)

db.init()

# ---------------------------------------------------------------------------
# Core film physics primitives
# ---------------------------------------------------------------------------

def lift_shadows(img, r=30, g=25, b=20):
    """Lift shadow floor (D-min / base fog density)."""
    bands = img.split()
    result = []
    for band, floor in zip(bands, (r, g, b)):
        lut = [floor + int(v * (255 - floor) / 255) for v in range(256)]
        result.append(band.point(lut))
    return Image.merge('RGB', result)


def apply_film_shoulder(img, start=0.76, strength=0.82):
    """
    Compress highlights with a smooth asymptotic rolloff — the 'shoulder'
    of a real H&D (Hurter-Driffield) film curve.
    Film never clips; it just rolls off gracefully.
    """
    lut = []
    for i in range(256):
        t = i / 255.0
        if t > start:
            excess = (t - start) / (1.0 - start)          # 0→1 in highlight zone
            compressed = 1.0 - np.exp(-excess * 2.8)       # asymptotic approach to 1
            t = start + compressed * (1.0 - start) * strength
        lut.append(int(np.clip(t * 255, 0, 255)))
    bands = img.split()
    return Image.merge('RGB', [b.point(lut) for b in bands])


def apply_film_toe(img, toe=0.08, lift=0.05):
    """
    Soft shadow toe: the darkest pixels compress slightly upward,
    simulating the toe region of the film curve.
    """
    lut = []
    for i in range(256):
        t = i / 255.0
        if t < toe:
            factor = t / toe
            t = lift * (1 - factor) + t * factor
        lut.append(int(np.clip(t * 255, 0, 255)))
    bands = img.split()
    return Image.merge('RGB', [b.point(lut) for b in bands])


def add_halation(img, strength=0.14, radius=14):
    """
    Halation: bright light bleeds through the emulsion and reflects off the
    film base, creating a warm glow around highlights.
    Color is warm (red > green >> blue) because the anti-halation layer
    absorbs blue/green more efficiently than red.
    """
    arr = np.array(img, dtype=np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # Threshold: only brightest highlights bleed
    highlights = np.clip((luma - 185) / 70.0, 0, 1) ** 1.4
    h_img = Image.fromarray((highlights * 255).astype(np.uint8))
    glow = np.array(h_img.filter(ImageFilter.GaussianBlur(radius)),
                    dtype=np.float32) / 255.0
    arr[:, :, 0] = np.clip(arr[:, :, 0] + glow * strength * 255 * 1.00, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + glow * strength * 255 * 0.22, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + glow * strength * 255 * 0.04, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def add_film_grain(img, intensity=18):
    """
    Organic film grain: silver halide crystals cluster spatially.
    Achieved by mixing fine per-pixel noise with a blurred (clustered) version,
    weighted by a midtone mask (grain is invisible in deep shadows/highlights).
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    midtone_mask = 4 * (luma / 255) * (1 - luma / 255)

    fine = np.random.normal(0, intensity, (h, w)).astype(np.float32)
    # Blur a copy to create spatial clumping
    fine_u8 = np.clip(fine + 128, 0, 255).astype(np.uint8)
    clustered = (np.array(Image.fromarray(fine_u8).filter(ImageFilter.GaussianBlur(1.8)),
                          dtype=np.float32) - 128)
    grain = fine * 0.60 + clustered * 0.40
    grain *= midtone_mask

    for i in range(3):
        arr[:, :, i] = np.clip(arr[:, :, i] + grain, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def add_vignette(img, strength=0.55, feather=0.40):
    """Optical vignette from lens falloff."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    y, x = np.mgrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    radius = np.sqrt(x ** 2 + y ** 2) / np.sqrt(2)
    vignette = np.clip((radius - feather) / (1 - feather), 0, 1)
    darkening = 1 - strength * vignette
    for i in range(3):
        arr[:, :, i] = np.clip(arr[:, :, i] * darkening, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def add_channel_blur(img, blue_radius=0.9):
    """
    Slight blue-channel softness: the blue-sensitive layer sits deepest in
    the emulsion stack, so it receives slightly diffused light.
    """
    r, g, b = img.split()
    b = b.filter(ImageFilter.GaussianBlur(blue_radius))
    return Image.merge('RGB', (r, g, b))


def _boost_saturation(arr, factor):
    luma = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1]
            + 0.114 * arr[:, :, 2])[:, :, np.newaxis]
    return np.clip(luma + (arr - luma) * factor, 0, 255)


def _s_curve_lut(strength=0.18):
    def curve(v):
        t = v / 255.0
        t = t + strength * t * (1 - t) * (2 * t - 1) * 2
        return int(np.clip(t * 255, 0, 255))
    return [curve(v) for v in range(256)]


# ---------------------------------------------------------------------------
# Style 1: Kodak Gold 200
# ---------------------------------------------------------------------------

def apply_kodak_gold(img):
    """
    Warm, saturated, orange-biased. Characteristic of holiday/consumer film.
    Kodak Gold has a strong toe, orange shadows, and open warm highlights.
    """
    img = img.convert('RGB')
    # Base fog: asymmetric warm lift
    img = lift_shadows(img, r=32, g=26, b=18)
    img = apply_film_toe(img, toe=0.10, lift=0.06)
    img = apply_film_shoulder(img, start=0.78, strength=0.80)

    arr = np.array(img, dtype=np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Warm cast — more red, slightly boosted green, pulled blue
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.09, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.02, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.90, 0, 255)

    # Shadow crossover: Kodak shadows shift orange-warm
    shadow = np.clip(1 - luma / 110, 0, 1)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + shadow * 14, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - shadow * 10, 0, 255)

    # Blue desaturation in sky/water
    r2, g2, b2 = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    blue_mask = ((b2 > r2) & (b2 > g2)).astype(np.float32) * 0.32
    luma2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
    for i in range(3):
        arr[:, :, i] = arr[:, :, i] * (1 - blue_mask) + luma2 * blue_mask

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = add_halation(img, strength=0.16, radius=16)
    img = add_channel_blur(img)
    img = add_film_grain(img, intensity=17)
    img = add_vignette(img, strength=0.52, feather=0.42)
    return img


# ---------------------------------------------------------------------------
# Style 2: Kodak Tri-X 400 B&W
# ---------------------------------------------------------------------------

def apply_trix_bw(img):
    """
    Punchy, gritty B&W. Strong midtone contrast, heavy clumping grain,
    pronounced halation around lights.
    """
    img = img.convert('RGB')
    arr = np.array(img, dtype=np.float32)
    # Panchromatic weights — Tri-X favours green slightly
    luma = 0.21 * arr[:, :, 0] + 0.72 * arr[:, :, 1] + 0.07 * arr[:, :, 2]
    arr[:, :, 0] = arr[:, :, 1] = arr[:, :, 2] = luma
    img = Image.fromarray(arr.astype(np.uint8))

    img = lift_shadows(img, r=10, g=10, b=10)
    img = apply_film_toe(img, toe=0.06, lift=0.03)
    img = apply_film_shoulder(img, start=0.80, strength=0.75)

    # Punchy midtone S-curve
    lut = _s_curve_lut(0.20)
    img = Image.merge('RGB', [b.point(lut) for b in img.split()])

    # Tri-X halation: grey-warm glow (B&W but base is slightly warm)
    arr2 = np.array(img, dtype=np.float32)
    luma2 = arr2[:, :, 0]
    highlights = np.clip((luma2 - 200) / 55.0, 0, 1) ** 1.2
    h_img = Image.fromarray((highlights * 255).astype(np.uint8))
    glow = np.array(h_img.filter(ImageFilter.GaussianBlur(12)),
                    dtype=np.float32) / 255.0
    arr2[:, :, 0] = np.clip(arr2[:, :, 0] + glow * 0.18 * 255, 0, 255)
    arr2[:, :, 1] = np.clip(arr2[:, :, 1] + glow * 0.15 * 255, 0, 255)
    arr2[:, :, 2] = np.clip(arr2[:, :, 2] + glow * 0.11 * 255, 0, 255)
    img = Image.fromarray(arr2.astype(np.uint8))

    img = add_film_grain(img, intensity=30)
    img = add_vignette(img, strength=0.62, feather=0.30)
    return img


# ---------------------------------------------------------------------------
# Style 3: Vintage 70s (Expired / Ektachrome-inspired)
# ---------------------------------------------------------------------------

def apply_vintage_70s(img):
    """
    Expired 70s slide film: milky shadows, heavy orange cast,
    faded whites, warm shadow crossover.
    """
    img = img.convert('RGB')
    img = lift_shadows(img, r=50, g=42, b=30)
    img = apply_film_toe(img, toe=0.18, lift=0.10)
    img = apply_film_shoulder(img, start=0.72, strength=0.70)

    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.13, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.04, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.72, 0, 255)
    # Fade whites
    arr = arr * (222 / 255)

    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # Warm shadow crossover
    shadow = np.clip(1 - luma / 90, 0, 1) * 0.45
    arr[:, :, 0] = np.clip(arr[:, :, 0] + shadow * 22, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - shadow * 14, 0, 255)
    # Blue desaturation
    r2, g2, b2 = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    blue_mask = ((b2 > r2) & (b2 > g2)).astype(np.float32) * 0.42
    luma2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
    for i in range(3):
        arr[:, :, i] = arr[:, :, i] * (1 - blue_mask) + luma2 * blue_mask

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = add_halation(img, strength=0.20, radius=20)
    img = add_channel_blur(img, blue_radius=1.2)
    img = add_film_grain(img, intensity=14)
    img = add_vignette(img, strength=0.36, feather=0.50)
    return img


# ---------------------------------------------------------------------------
# Style 4: Cinematic (teal shadows / orange highlights)
# ---------------------------------------------------------------------------

def apply_cinematic(img):
    """
    Hollywood DI grade. Clean shoulder, precise split tone,
    subtle halation around practicals.
    """
    img = img.convert('RGB')
    img = lift_shadows(img, r=18, g=16, b=14)
    img = apply_film_shoulder(img, start=0.82, strength=0.78)

    arr = np.array(img, dtype=np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    shadow    = np.clip(1 - luma / 120, 0, 1)
    highlight = np.clip((luma - 135) / 120, 0, 1)
    arr[:, :, 0] -= shadow * 18
    arr[:, :, 1] += shadow * 6
    arr[:, :, 2] += shadow * 16
    arr[:, :, 0] += highlight * 20
    arr[:, :, 1] += highlight * 8
    arr[:, :, 2] -= highlight * 22

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = add_halation(img, strength=0.10, radius=12)
    img = add_channel_blur(img)
    img = add_film_grain(img, intensity=19)
    img = add_vignette(img, strength=0.58, feather=0.35)
    return img


# ---------------------------------------------------------------------------
# Style 5: Fuji Velvia 50
# ---------------------------------------------------------------------------

def apply_fuji_velvia(img):
    """
    Hyper-saturated reversal film. Vivid greens and reds, cool neutrals,
    almost no grain (ISO 50), very sharp (minimal channel blur).
    """
    img = img.convert('RGB')
    img = lift_shadows(img, r=12, g=10, b=16)
    img = apply_film_toe(img, toe=0.07, lift=0.04)
    img = apply_film_shoulder(img, start=0.80, strength=0.72)

    arr = np.array(img, dtype=np.float32)
    arr = _boost_saturation(arr, 1.50)
    # Velvia: greens especially vivid, slight cool overall
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.04, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.11, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.04, 0, 255)
    # Cool shadow crossover (Velvia shadows go slightly cyan)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    shadow = np.clip(1 - luma / 80, 0, 1)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + shadow * 10, 0, 255)
    arr[:, :, 0] = np.clip(arr[:, :, 0] - shadow * 6, 0, 255)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    lut = _s_curve_lut(0.14)
    img = Image.merge('RGB', [b.point(lut) for b in img.split()])
    # No channel blur — Velvia is razor-sharp
    img = add_film_grain(img, intensity=5)
    img = add_vignette(img, strength=0.42, feather=0.48)
    return img


# ---------------------------------------------------------------------------
# Style 6: Kodak Portra 400
# ---------------------------------------------------------------------------

def apply_portra_400(img):
    """
    Neutral, open, creamy. Portra is famous for beautiful skin tones,
    very open highlights, and a subtle cool-shadow / warm-highlight split tone.
    """
    img = img.convert('RGB')
    img = lift_shadows(img, r=20, g=18, b=22)   # neutral-to-cool lift
    img = apply_film_toe(img, toe=0.10, lift=0.04)
    img = apply_film_shoulder(img, start=0.74, strength=0.88)  # very open highlights

    arr = np.array(img, dtype=np.float32)
    arr = _boost_saturation(arr, 0.86)   # slightly desaturated (natural look)

    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    highlight = np.clip((luma - 128) / 128, 0, 1)
    shadow    = np.clip(1 - luma / 100, 0, 1)
    # Warm highlights, cool-green shadows (Portra crossover)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + highlight * 9, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - highlight * 5, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + shadow * 5, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + shadow * 8, 0, 255)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = add_halation(img, strength=0.13, radius=18)   # beautiful soft halation
    img = add_channel_blur(img, blue_radius=1.0)
    img = add_film_grain(img, intensity=11)
    img = add_vignette(img, strength=0.30, feather=0.55)
    return img


# ---------------------------------------------------------------------------
# Style 7: Ilford HP5 B&W
# ---------------------------------------------------------------------------

def apply_ilford_hp5(img):
    """
    Softer B&W than Tri-X. More latitude, open shadows, medium grain.
    HP5 is a workhorse film — versatile, less dramatic than Tri-X.
    """
    img = img.convert('RGB')
    arr = np.array(img, dtype=np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    arr[:, :, 0] = arr[:, :, 1] = arr[:, :, 2] = luma
    img = Image.fromarray(arr.astype(np.uint8))

    img = lift_shadows(img, r=18, g=18, b=18)
    img = apply_film_toe(img, toe=0.10, lift=0.05)
    img = apply_film_shoulder(img, start=0.82, strength=0.80)

    lut = _s_curve_lut(0.11)   # gentle S — less punchy than Tri-X
    img = Image.merge('RGB', [b.point(lut) for b in img.split()])

    # HP5 halation: subtle, neutral-grey
    arr2 = np.array(img, dtype=np.float32)
    luma2 = arr2[:, :, 0]
    highlights = np.clip((luma2 - 205) / 50.0, 0, 1)
    h_img = Image.fromarray((highlights * 255).astype(np.uint8))
    glow = np.array(h_img.filter(ImageFilter.GaussianBlur(10)),
                    dtype=np.float32) / 255.0
    for i in range(3):
        arr2[:, :, i] = np.clip(arr2[:, :, i] + glow * 0.12 * 255, 0, 255)
    img = Image.fromarray(arr2.astype(np.uint8))

    img = add_film_grain(img, intensity=22)
    img = add_vignette(img, strength=0.44, feather=0.46)
    return img


# ---------------------------------------------------------------------------
# Optional overlay effects
# ---------------------------------------------------------------------------

def add_light_leak(img):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cx = float(np.random.choice([0, w]))
    cy = float(np.random.choice([0, h]))
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2) * 0.62
    leak = np.clip(1 - dist / max_dist, 0, 1) ** 1.6 * 0.68
    colors = [(255, 90, 10), (240, 55, 140), (255, 190, 25)]
    cr, cg, cb = colors[np.random.randint(0, 3)]
    arr[:, :, 0] = np.clip(arr[:, :, 0] + leak * cr * 0.80, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + leak * cg * 0.38, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + leak * cb * 0.18, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_date_stamp(img, date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime('%m %d %Y')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.load_default(size=max(14, h // 22))
    except TypeError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), date_str, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = max(10, h // 55)
    draw.text((w - tw - margin, h - th - margin), date_str,
              fill=(255, 100, 0), font=font)
    return img


def add_film_border(img):
    w, h = img.size
    bv = max(22, int(h * 0.058))
    bh = max(10, int(w * 0.022))
    out = Image.new('RGB', (w + bh * 2, h + bv * 2), (8, 8, 8))
    out.paste(img, (bh, bv))
    draw = ImageDraw.Draw(out)
    new_w, new_h = out.size
    hole_r = max(5, bv // 3)
    spacing = hole_r * 5
    x = spacing
    while x < new_w - spacing // 2:
        for cy in [bv // 2, new_h - bv // 2]:
            draw.ellipse([x - hole_r, cy - hole_r, x + hole_r, cy + hole_r],
                         fill=(1, 1, 1), outline=(30, 30, 30))
        x += spacing
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

STYLES = {
    'kodak-gold':  (apply_kodak_gold,  'kodak_gold.jpg'),
    'trix-bw':     (apply_trix_bw,     'trix_bw.jpg'),
    'vintage-70s': (apply_vintage_70s, 'vintage_70s.jpg'),
    'cinematic':   (apply_cinematic,   'cinematic.jpg'),
    'fuji-velvia': (apply_fuji_velvia, 'fuji_velvia.jpg'),
    'portra-400':  (apply_portra_400,  'portra_400.jpg'),
    'ilford-hp5':  (apply_ilford_hp5,  'ilford_hp5.jpg'),
}

STYLE_NAMES = {
    'kodak-gold':  'Kodak Gold',
    'trix-bw':     'Tri-X B&W',
    'vintage-70s': 'Vintage 70s',
    'cinematic':   'Cinematic',
    'fuji-velvia': 'Fuji Velvia',
    'portra-400':  'Portra 400',
    'ilford-hp5':  'Ilford HP5',
}


def apply_filter(img, style='kodak-gold', intensity=100,
                 light_leak=False, date_stamp=False, film_border=False):
    original = img.convert('RGB')
    fn, _ = STYLES.get(style, STYLES['kodak-gold'])
    result = fn(img)

    if intensity < 100:
        orig_arr   = np.array(original,  dtype=np.float32)
        result_arr = np.array(result,    dtype=np.float32)
        alpha = intensity / 100.0
        result = Image.fromarray(
            (orig_arr * (1 - alpha) + result_arr * alpha).astype(np.uint8))

    if light_leak:
        result = add_light_leak(result)
    if date_stamp:
        result = add_date_stamp(result)
    if film_border:
        result = add_film_border(result)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_album_id():
    """Return 6 uppercase alphanumeric characters."""
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(6))


def _parse_request():
    if 'image' not in request.files:
        return None, None, None, ('No image field', 400)
    file = request.files['image']
    if file.filename == '':
        return None, None, None, ('No file selected', 400)
    style = request.form.get('style', 'kodak-gold')
    if style not in STYLES:
        return None, None, None, ('Unknown style', 400)
    opts = {
        'intensity':   int(request.form.get('intensity', 100)),
        'light_leak':  request.form.get('light_leak')  == '1',
        'date_stamp':  request.form.get('date_stamp')  == '1',
        'film_border': request.form.get('film_border') == '1',
    }
    return file, style, opts, None


# ---------------------------------------------------------------------------
# Routes — new collaborative album feature
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/create', methods=['POST'])
def create_album():
    album_name = request.form.get('album_name', '').strip()
    user_name  = request.form.get('user_name', '').strip()
    if not album_name or not user_name:
        return jsonify({'error': 'album_name and user_name are required'}), 400
    album_id   = make_album_id()
    created_at = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    db.create_album(album_id, album_name, created_at)
    # Create album photo directory
    album_photo_dir = os.path.join(ALBUMS_DIR, album_id)
    os.makedirs(album_photo_dir, exist_ok=True)
    return jsonify({'id': album_id, 'name': album_name})


@app.route('/a/<album_id>')
def view_album(album_id):
    album_row = db.get_album(album_id)
    if album_row is None:
        abort(404, description='Album not found')
    album = {'id': album_row['id'], 'name': album_row['name']}
    return render_template('album.html', album=album)


@app.route('/a/<album_id>/qr')
def album_qr(album_id):
    album_row = db.get_album(album_id)
    if album_row is None:
        abort(404, description='Album not found')
    url = f'{request.host_url}a/{album_id}'
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=6,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    # Add 20px white border padding
    w, h = qr_img.size
    padded = Image.new('RGB', (w + 40, h + 40), (255, 255, 255))
    padded.paste(qr_img, (20, 20))
    buf = BytesIO()
    padded.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/a/<album_id>/exists')
def album_exists(album_id):
    album_row = db.get_album(album_id)
    return jsonify({'exists': album_row is not None})


@app.route('/a/<album_id>/upload', methods=['POST'])
def album_upload(album_id):
    album_row = db.get_album(album_id)
    if album_row is None:
        abort(404, description='Album not found')

    if 'image' not in request.files:
        return jsonify({'error': 'No image field'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    style = request.form.get('style', 'kodak-gold')
    if style not in STYLES:
        return jsonify({'error': 'Unknown style'}), 400

    uploaded_by  = request.form.get('uploaded_by', 'Anonymous').strip() or 'Anonymous'
    intensity    = int(request.form.get('intensity', 100))
    light_leak   = request.form.get('light_leak', '0') == '1'
    date_stamp   = request.form.get('date_stamp', '0') == '1'
    film_border  = request.form.get('film_border', '0') == '1'

    original = Image.open(BytesIO(file.read())).convert('RGB')
    result   = apply_filter(original, style, intensity=intensity,
                            light_leak=light_leak, date_stamp=date_stamp,
                            film_border=film_border)

    ts       = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f'{ts}_{style}.jpg'

    album_photo_dir = os.path.join(ALBUMS_DIR, album_id)
    os.makedirs(album_photo_dir, exist_ok=True)
    result.save(os.path.join(album_photo_dir, filename), format='JPEG', quality=92)

    uploaded_at = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    photo_id    = db.add_photo(album_id, filename, style, uploaded_by, uploaded_at)

    # Detect faces & compute embeddings from the original (pre-filter) image
    _detect_faces(photo_id, album_id, filename, pil_img=original)

    photo_url = f'/static/albums/{album_id}/{filename}'
    return jsonify({
        'id':          photo_id,
        'url':         photo_url,
        'filename':    filename,
        'style':       style,
        'uploaded_by': uploaded_by,
        'uploaded_at': uploaded_at,
        'tags':        [],
    })


@app.route('/a/<album_id>/photos')
def album_photos(album_id):
    album_row = db.get_album(album_id)
    if album_row is None:
        abort(404, description='Album not found')
    since = int(request.args.get('since', 0))
    photos = db.get_photos(album_id, since_id=since)
    return jsonify(photos)


@app.route('/a/<album_id>/tag', methods=['POST'])
def album_tag(album_id):
    if db.get_album(album_id) is None:
        abort(404)
    data     = request.get_json(force=True, silent=True) or {}
    photo_id = data.get('photo_id')
    name     = (data.get('name') or '').strip()
    face_id  = data.get('face_id')        # optional — which face was tagged
    if not photo_id or not name:
        return jsonify({'error': 'photo_id and name are required'}), 400
    db.add_tag(photo_id, name, face_id=face_id)
    return jsonify({'ok': True, 'tags': db.get_tags(photo_id)})


def _detect_faces(photo_id, album_id, filename, pil_img=None):
    """Run SCRFD detection + ArcFace embedding, save crops, store in DB."""
    if pil_img is None:
        photo_path = os.path.join(ALBUMS_DIR, album_id, filename)
        if not os.path.isfile(photo_path):
            db.mark_faces_detected(photo_id)
            return []
        pil_img = Image.open(photo_path).convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    detections = _detect_faces_in_img(pil_img)

    faces_dir = os.path.join(ALBUMS_DIR, album_id, 'faces')
    os.makedirs(faces_dir, exist_ok=True)

    result = []
    for i, (bbox, kps) in enumerate(detections):
        crop      = _face_crop(pil_img, bbox, kps, size=88)
        fname     = f'face_{photo_id}_{i}.jpg'
        crop.save(os.path.join(faces_dir, fname), 'JPEG', quality=85)
        url       = f'/static/albums/{album_id}/faces/{fname}'
        embedding = _arcface_embed(crop)
        face_id   = db.add_face(photo_id, url, embedding=embedding)
        result.append({'id': face_id, 'crop_url': url})

    db.mark_faces_detected(photo_id)
    return result


@app.route('/a/<album_id>/photos/<int:photo_id>/faces')
def album_faces(album_id, photo_id):
    if db.get_album(album_id) is None:
        abort(404)
    if db.faces_are_detected(photo_id):
        raw = db.get_faces(photo_id)
        return jsonify({'faces': [{'id': f['id'], 'crop_url': f['crop_url']} for f in raw]})
    photos = db.get_photos(album_id)
    photo  = next((p for p in photos if p['id'] == photo_id), None)
    if photo is None:
        abort(404)
    faces = _detect_faces(photo_id, album_id, photo['filename'])
    return jsonify({'faces': faces})


def _search_embedding(q_emb, album_id, always_include_photo_id=None):
    """Return set of photo_ids whose faces are similar to q_emb (ArcFace cosine)."""
    all_faces = db.get_all_album_faces(album_id)
    THRESHOLD = 0.35   # ArcFace cosine sim: same person ≈ 0.4–0.9, different < 0.2
    matched = set()
    if always_include_photo_id is not None:
        matched.add(always_include_photo_id)
    for face in all_faces:
        raw = face['embedding']
        if not raw or len(raw) != _EMB_BYTES:
            continue
        emb = np.frombuffer(bytes(raw), dtype=np.float32)
        if float(np.dot(q_emb, emb)) >= THRESHOLD:
            matched.add(face['photo_id'])
    return matched


@app.route('/a/<album_id>/search-by-face/<int:face_id>')
def search_by_face(album_id, face_id):
    if db.get_album(album_id) is None:
        abort(404)
    query_face = db.get_face(face_id)
    if query_face is None or query_face['embedding'] is None:
        return jsonify({'crop_url': None, 'photo_ids': []})

    q_emb   = np.frombuffer(bytes(query_face['embedding']), dtype=np.float32)
    matched = _search_embedding(q_emb, album_id,
                                always_include_photo_id=query_face['photo_id'])
    return jsonify({
        'crop_url':  query_face['crop_url'],
        'photo_ids': list(matched),
    })


@app.route('/a/<album_id>/search-by-uploaded-face', methods=['POST'])
def search_by_uploaded_face(album_id):
    """Detect the face in a user-uploaded photo and search the album for it."""
    if db.get_album(album_id) is None:
        abort(404)
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    det_sess, _ = _get_sessions()
    if det_sess is None:
        return jsonify({'error': 'Face detection not available'}), 503

    pil_img    = Image.open(BytesIO(request.files['image'].read())).convert('RGB')
    detections = _detect_faces_in_img(pil_img)

    if not detections:
        return jsonify({'error': 'No face detected in your photo. Try a clearer front-facing shot.'}), 422

    # Use the largest detected face
    largest_bbox, largest_kps = max(detections, key=lambda d: (d[0][2]-d[0][0]) * (d[0][3]-d[0][1]))
    crop  = _face_crop(pil_img, largest_bbox, largest_kps, size=88)
    emb   = _arcface_embed(crop)
    if emb is None:
        return jsonify({'error': 'Could not compute face embedding'}), 500
    q_emb   = np.frombuffer(emb, dtype=np.float32)
    matched = _search_embedding(q_emb, album_id)

    # Return the crop as a data URL so the browser can show it in the filter bar
    import base64
    buf = BytesIO()
    crop.save(buf, 'JPEG', quality=85)
    crop_data_url = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        'crop_data_url': crop_data_url,
        'photo_ids':     list(matched),
    })


@app.route('/a/<album_id>/delete/<int:photo_id>', methods=['POST'])
def album_delete_photo(album_id, photo_id):
    album_row = db.get_album(album_id)
    if album_row is None:
        abort(404, description='Album not found')
    filename = db.delete_photo(photo_id, album_id)
    if filename:
        fp = os.path.join(ALBUMS_DIR, album_id, filename)
        if os.path.isfile(fp):
            os.remove(fp)
    return ('', 204)


# ---------------------------------------------------------------------------
# Legacy routes (kept for backward compatibility)
# ---------------------------------------------------------------------------

@app.route('/upload', methods=['POST'])
def upload():
    file, style, opts, err = _parse_request()
    if err:
        return err
    img    = Image.open(BytesIO(file.read()))
    result = apply_filter(img, style, **opts)
    _, dl_name = STYLES[style]
    buf = BytesIO()
    result.save(buf, format='JPEG', quality=92)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg', download_name=dl_name)


@app.route('/save', methods=['POST'])
def save():
    file, style, opts, err = _parse_request()
    if err:
        return err
    img    = Image.open(BytesIO(file.read()))
    result = apply_filter(img, style, **opts)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{ts}_{style}.jpg'
    result.save(os.path.join(ALBUM_DIR, filename), format='JPEG', quality=92)
    return jsonify({'url': f'/static/album/{filename}', 'filename': filename})


@app.route('/album')
def album():
    photos = []
    for fname in sorted(os.listdir(ALBUM_DIR), reverse=True):
        if not fname.endswith('.jpg'):
            continue
        parts = fname[:-4].split('_', 2)
        style_id = parts[2] if len(parts) == 3 else ''
        try:
            dt = datetime.strptime(parts[0] + parts[1], '%Y%m%d%H%M%S')
            date_str = dt.strftime('%b %d, %Y · %H:%M')
        except ValueError:
            date_str = ''
        photos.append({
            'filename': fname,
            'url':      f'/static/album/{fname}',
            'style':    STYLE_NAMES.get(style_id, style_id),
            'date':     date_str,
        })
    return render_template('gallery.html', photos=photos)


@app.route('/album/delete/<filename>', methods=['POST'])
def delete_photo(filename):
    if any(c in filename for c in ('/', '\\', '..')):
        return 'Invalid filename', 400
    fp = os.path.join(ALBUM_DIR, filename)
    if os.path.isfile(fp):
        os.remove(fp)
    return ('', 204)


if __name__ == '__main__':
    app.run(debug=True)
