import numpy as np
from io import BytesIO
from PIL import Image

_hald_cache = {}


def parse_hald_png(data):
    """
    Parse a Hald CLUT PNG into the same (N,N,N,3) float32 format as parse_cube.
    data: bytes or file-like object.
    Returns (grid_size, lut_3d) with axes [B, G, R].
    """
    img = Image.open(BytesIO(data) if isinstance(data, bytes) else data).convert('RGB')
    w, h = img.size
    if w != h:
        raise ValueError(f'Hald CLUT must be square, got {w}×{h}')
    # Image side length = L³ where L is the Hald level, N = L²
    L = round(w ** (1 / 3))
    if L ** 3 != w:
        raise ValueError(f'Cannot determine Hald CLUT level from image size {w}')
    N = L * L  # grid entries per channel

    arr = np.array(img, dtype=np.float32) / 255.0  # (w, w, 3)
    lut_3d = np.zeros((N, N, N, 3), dtype=np.float32)
    for b in range(N):
        x0 = (b % L) * N
        y0 = (b // L) * N
        # arr[y0:y0+N, x0:x0+N] is indexed [g, r] → lut_3d[b, g, r]
        lut_3d[b] = arr[y0:y0 + N, x0:x0 + N]
    return N, lut_3d


def load_hald_png(png_path):
    """Load and cache a Hald CLUT PNG from disk. Returns the lut_3d array."""
    if png_path not in _hald_cache:
        with open(png_path, 'rb') as f:
            _, lut_3d = parse_hald_png(f.read())
        _hald_cache[png_path] = lut_3d
    return _hald_cache[png_path]


def parse_cube(file_bytes):
    """Parse a .cube LUT file. Returns (size, lut_3d) where lut_3d is float32 (N,N,N,3)."""
    text = file_bytes.decode('utf-8', errors='ignore')
    size = None
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        upper = line.upper()
        if upper.startswith('LUT_3D_SIZE'):
            size = int(line.split()[-1])
            continue
        if upper.startswith('DOMAIN') or upper.startswith('TITLE') or upper.startswith('LUT_1D'):
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                entries.append([float(p) for p in parts])
            except ValueError:
                continue

    if size is None:
        raise ValueError('Missing LUT_3D_SIZE in .cube file')
    expected = size ** 3
    if len(entries) != expected:
        raise ValueError(f'Expected {expected} LUT entries for size {size}, got {len(entries)}')

    # Standard .cube axis order: R changes fastest, then G, then B
    arr = np.array(entries, dtype=np.float32).reshape(size, size, size, 3)
    return size, arr


def apply_lut(img_array, lut_3d, strength=1.0):
    """
    Apply a 3D LUT to an RGB image via trilinear interpolation.
    img_array: uint8 H×W×3 numpy array
    lut_3d: float32 (N,N,N,3) array from parse_cube, axes [B,G,R]
    strength: 0.0–1.0 blend factor
    Returns: uint8 H×W×3
    """
    N = lut_3d.shape[0]
    orig = img_array.astype(np.float32)

    # Scale pixel values to LUT grid coordinates
    scaled = orig / 255.0 * (N - 1)
    r_s = scaled[..., 0]
    g_s = scaled[..., 1]
    b_s = scaled[..., 2]

    r0 = np.clip(np.floor(r_s).astype(np.int32), 0, N - 2)
    g0 = np.clip(np.floor(g_s).astype(np.int32), 0, N - 2)
    b0 = np.clip(np.floor(b_s).astype(np.int32), 0, N - 2)

    fr = (r_s - r0)[..., np.newaxis]
    fg = (g_s - g0)[..., np.newaxis]
    fb = (b_s - b0)[..., np.newaxis]

    # Sample 8 corners of the LUT cube (B,G,R indexing)
    c000 = lut_3d[b0,     g0,     r0    ]
    c001 = lut_3d[b0,     g0,     r0 + 1]
    c010 = lut_3d[b0,     g0 + 1, r0    ]
    c011 = lut_3d[b0,     g0 + 1, r0 + 1]
    c100 = lut_3d[b0 + 1, g0,     r0    ]
    c101 = lut_3d[b0 + 1, g0,     r0 + 1]
    c110 = lut_3d[b0 + 1, g0 + 1, r0    ]
    c111 = lut_3d[b0 + 1, g0 + 1, r0 + 1]

    result = (c000 * (1 - fr) * (1 - fg) * (1 - fb) +
              c001 * fr       * (1 - fg) * (1 - fb) +
              c010 * (1 - fr) * fg       * (1 - fb) +
              c011 * fr       * fg       * (1 - fb) +
              c100 * (1 - fr) * (1 - fg) * fb       +
              c101 * fr       * (1 - fg) * fb       +
              c110 * (1 - fr) * fg       * fb       +
              c111 * fr       * fg       * fb)

    result = np.clip(result * 255.0, 0, 255)

    if strength < 1.0:
        result = orig * (1.0 - strength) + result * strength

    return result.astype(np.uint8)
