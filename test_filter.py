"""Smoke tests for all 4 film styles."""
import sys
import numpy as np
from PIL import Image
from io import BytesIO

sys.path.insert(0, '.')
from app import lift_shadows, apply_filter, STYLES


def make_gradient():
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            arr[y, x] = [x, y, (x + y) // 2]
    arr[:40, :] = [80, 100, 200]  # blue sky strip
    return Image.fromarray(arr)


def test_shadow_lift():
    black = Image.new('RGB', (50, 50), (0, 0, 0))
    lifted = lift_shadows(black, r=30, g=25, b=20)
    arr = np.array(lifted)
    assert arr[:, :, 0].min() >= 30
    assert arr[:, :, 1].min() >= 25
    assert arr[:, :, 2].min() >= 20
    print("PASS shadow_lift")


def test_all_styles():
    img = make_gradient()
    for style in STYLES:
        result = apply_filter(img, style)
        assert result.mode == 'RGB', f"{style}: bad mode"
        assert result.size == img.size, f"{style}: size mismatch"
        buf = BytesIO()
        result.save(buf, format='JPEG')
        assert buf.tell() > 500, f"{style}: output too small"
        print(f"PASS {style} — {buf.tell()} bytes")


if __name__ == '__main__':
    test_shadow_lift()
    test_all_styles()
    print("All tests passed.")
