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

_mp_detector = None
_MP_MODEL    = os.path.join(os.path.dirname(__file__), 'blaze_face.tflite')


def _get_detector():
    global _mp_detector
    if _mp_detector is None:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_tasks
            base = mp_tasks.BaseOptions(model_asset_path=_MP_MODEL)
            opts = vision.FaceDetectorOptions(
                base_options=base,
                min_detection_confidence=0.45,
            )
            _mp_detector = vision.FaceDetector.create_from_options(opts)
        except Exception:
            pass
    return _mp_detector


def _mp_detect(pil_img):
    """Return raw MediaPipe detection objects for pil_img."""
    detector = _get_detector()
    if detector is None:
        return []
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_img))
    return detector.detect(mp_image).detections


def _aligned_crop(pil_img, detection, size=88):
    """
    Return a size×size face crop aligned using the eye keypoints.
    Aligning to a canonical upright pose before embedding makes the
    same person recognisable across different head tilts.
    """
    import math
    iw, ih = pil_img.size
    kps    = detection.keypoints   # [left_eye, right_eye, nose, mouth, left_ear, right_ear]

    if len(kps) >= 2:
        le_x, le_y = kps[0].x * iw, kps[0].y * ih   # left eye  (in image coords)
        re_x, re_y = kps[1].x * iw, kps[1].y * ih   # right eye
        angle      = math.degrees(math.atan2(re_y - le_y, re_x - le_x))
        eye_cx     = (le_x + re_x) / 2
        eye_cy     = (le_y + re_y) / 2
        # Rotate the whole image so the eye line is horizontal
        work = pil_img.rotate(-angle, resample=Image.BICUBIC,
                              center=(eye_cx, eye_cy), expand=False)
    else:
        work = pil_img

    bb  = detection.bounding_box
    x, y, w, h = bb.origin_x, bb.origin_y, bb.width, bb.height
    pad = int(max(w, h) * 0.28)
    crop = work.crop((
        max(0, x - pad),  max(0, y - pad),
        min(iw, x + w + pad), min(ih, y + h + pad),
    )).resize((size, size), Image.LANCZOS)
    return crop

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


_LBP_NEIGHBORS = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
_LBP_CELLS     = 7
_LBP_BINS      = 64
_EMB_BYTES     = _LBP_CELLS * _LBP_CELLS * _LBP_BINS * 4   # float32 bytes


def _compute_embedding(crop_pil):
    """
    LBP (Local Binary Pattern) histogram embedding.
    Much more robust to lighting variation than raw pixels.
    Produces a 7×7 grid of 64-bin histograms = 3136-dim unit vector.
    """
    # Resize to 64×64 and apply gamma normalisation for illumination robustness
    gray = np.array(crop_pil.convert('L').resize((64, 64), Image.LANCZOS),
                    dtype=np.float32)
    gray = (np.sqrt(gray / 255.0) * 255).astype(np.uint8)

    center = gray[1:63, 1:63].astype(np.int16)   # 62×62 centre region
    lbp    = np.zeros((62, 62), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_LBP_NEIGHBORS):
        nbr  = gray[1+dy:63+dy, 1+dx:63+dx].astype(np.int16)
        lbp |= (nbr >= center).astype(np.uint8) << bit

    # Spatial histogram over 7×7 grid
    cs = 62 // _LBP_CELLS   # cell size in pixels (= 8)
    features = []
    for r in range(_LBP_CELLS):
        for c in range(_LBP_CELLS):
            cell = lbp[r*cs:(r+1)*cs, c*cs:(c+1)*cs]
            hist, _ = np.histogram(cell.ravel(), bins=_LBP_BINS, range=(0, 256))
            features.append(hist.astype(np.float32))

    arr  = np.concatenate(features)           # 3136-dim
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr.tobytes()



def _detect_faces(photo_id, album_id, filename, pil_img=None):
    """Run MediaPipe face detection, save crops, store in DB. Returns face list.

    Pass pil_img to detect on the original pre-filter image.
    Omit to fall back to reading the saved file from disk.
    """
    if pil_img is None:
        photo_path = os.path.join(ALBUMS_DIR, album_id, filename)
        if not os.path.isfile(photo_path):
            db.mark_faces_detected(photo_id)
            return []
        pil_img = Image.open(photo_path).convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    # Downscale for speed; MediaPipe works on full-res but 1200px is enough
    iw, ih  = pil_img.size
    scale   = min(1200 / max(iw, ih), 1.0)
    work    = pil_img if scale == 1.0 else pil_img.resize(
                  (int(iw * scale), int(ih * scale)), Image.LANCZOS)

    detections = _mp_detect(work)

    faces_dir = os.path.join(ALBUMS_DIR, album_id, 'faces')
    os.makedirs(faces_dir, exist_ok=True)

    result = []
    for i, det in enumerate(detections):
        # Aligned crop on the work-size image; embedding computed from aligned face
        crop      = _aligned_crop(work, det, size=88)
        fname     = f'face_{photo_id}_{i}.jpg'
        crop.save(os.path.join(faces_dir, fname), 'JPEG', quality=85)
        url       = f'/static/albums/{album_id}/faces/{fname}'
        embedding = _compute_embedding(crop)
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
    """Return set of photo_ids whose faces are similar to q_emb."""
    all_faces = db.get_all_album_faces(album_id)
    THRESHOLD = 0.88   # LBP cosine similarity; same person ≈ 0.90+, different ≈ 0.70
    matched = set()
    if always_include_photo_id is not None:
        matched.add(always_include_photo_id)
    for face in all_faces:
        raw = face['embedding']
        if not raw or len(raw) != _EMB_BYTES:   # skip old/incompatible embeddings
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

    if _get_detector() is None:
        return jsonify({'error': 'Face detection not available'}), 503

    pil_img = Image.open(BytesIO(request.files['image'].read())).convert('RGB')
    iw, ih  = pil_img.size
    scale   = min(1200 / max(iw, ih), 1.0)
    work    = pil_img if scale == 1.0 else pil_img.resize(
                  (int(iw * scale), int(ih * scale)), Image.LANCZOS)
    detections = _mp_detect(work)

    if not detections:
        return jsonify({'error': 'No face detected in your photo. Try a clearer front-facing shot.'}), 422

    # Use the largest detected face
    largest = max(detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    crop    = _aligned_crop(work, largest, size=88)

    q_emb = np.frombuffer(_compute_embedding(crop), dtype=np.float32)
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
