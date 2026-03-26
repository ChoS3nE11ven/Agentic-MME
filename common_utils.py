from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Increase PIL's decompression bomb limit to handle large images
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 300_000_000  # 300M pixels (default is ~89M)
except Exception:
    pass

_IMG_RE = re.compile(r"^transformed_image_(\d+)\.png$", re.IGNORECASE)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def get_adaptive_image_params(image_count: int) -> dict:
    """
    Get adaptive compression parameters based on how many images are in the conversation.
    More images = more aggressive compression to avoid 413 Payload Too Large.
    
    Args:
        image_count: Number of images already in the conversation
    
    Returns:
        dict with max_size_mb, max_pixels, quality
    """
    if image_count <= 2:
        # Few images: high quality
        return {"max_size_mb": 5.0, "max_pixels": 2048 * 2048, "quality": 95}
    elif image_count <= 5:
        # Moderate: medium quality
        return {"max_size_mb": 3.0, "max_pixels": 1536 * 1536, "quality": 85}
    elif image_count <= 8:
        # Many images: lower quality
        return {"max_size_mb": 2.0, "max_pixels": 1280 * 1280, "quality": 75}
    elif image_count <= 12:
        # Lots of images: aggressive compression
        return {"max_size_mb": 1.5, "max_pixels": 1024 * 1024, "quality": 65}
    else:
        # Very many images: very aggressive
        return {"max_size_mb": 1.0, "max_pixels": 800 * 800, "quality": 55}


def image_to_data_url(image_path: Path, max_pixels: int = 2048 * 2048, quality: int = 95, max_size_mb: float = 15.0) -> str:
    """
    Convert image to data URL with automatic resizing for large images.
    
    Args:
        image_path: Path to the image file
        max_pixels: Maximum total pixels (width * height). Images larger than this will be resized.
                   Default: 2048*2048 = 4,194,304 pixels (~4MP)
        quality: JPEG quality for resized images (1-100). Default: 95
        max_size_mb: Maximum file size in MB before compression. Default: 15MB
                    (OpenAI API has ~20MB limit, we use 15MB to be safe after base64 encoding)
    
    Returns:
        Data URL string
    """
    from PIL import Image
    import io
    
    # Check file size first
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    
    # Open image to check size
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height
    
    # If image is small enough in both pixels and file size, use original file
    if total_pixels <= max_pixels and file_size_mb <= max_size_mb:
        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        # Detect format from file extension
        ext = image_path.suffix.lower()
        mime = "image/png" if ext == ".png" else f"image/{ext[1:]}"
        return f"data:{mime};base64,{b64}"
    
    # Need to resize/compress
    # Calculate scale based on pixels
    if total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        img_resized = img
        new_width, new_height = width, height
    
    # Convert to RGB if necessary (for JPEG)
    if img_resized.mode in ('RGBA', 'LA', 'P'):
        # Create white background
        background = Image.new('RGB', img_resized.size, (255, 255, 255))
        if img_resized.mode == 'P':
            img_resized = img_resized.convert('RGBA')
        if 'A' in img_resized.mode:
            background.paste(img_resized, mask=img_resized.split()[-1])
        else:
            background.paste(img_resized)
        img_resized = background
    elif img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')
    
    # Try different quality levels to get under size limit
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    quality_levels = [quality, 90, 85, 80, 75, 70, 60, 50]
    
    for q in quality_levels:
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG', quality=q, optimize=True)
        data = buffer.getvalue()
        
        if len(data) <= max_size_bytes:
            if q < quality:
                print(f"[Image] Compressed {image_path.name}: {file_size_mb:.1f}MB -> {len(data)/1024/1024:.1f}MB (quality={q})")
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    
    # Still too large, need to resize more aggressively
    for scale_factor in [0.8, 0.6, 0.5, 0.4]:
        smaller_width = int(new_width * scale_factor)
        smaller_height = int(new_height * scale_factor)
        img_smaller = img_resized.resize((smaller_width, smaller_height), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img_smaller.save(buffer, format='JPEG', quality=60, optimize=True)
        data = buffer.getvalue()
        
        if len(data) <= max_size_bytes:
            print(f"[Image] Resized {image_path.name}: {file_size_mb:.1f}MB -> {len(data)/1024/1024:.1f}MB (scale={scale_factor})")
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
    
    # Last resort: use whatever we have
    print(f"[Image] Warning: {image_path.name} still large after compression: {len(data)/1024/1024:.1f}MB")
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def list_transformed_pngs(dir_: Path) -> list[Path]:
    if not dir_.exists():
        return []
    files = [p for p in dir_.iterdir() if p.is_file() and _IMG_RE.match(p.name)]
    def key(p: Path) -> int:
        m = _IMG_RE.match(p.name)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

def make_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_config: Optional[Path] = None,
) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package missing. Install: pip install openai>=1.46.1")

    if api_config is not None and api_config.exists():
        cfg = read_json(api_config)
        api_key = api_key or cfg.get("api_key")
        base_url = base_url or cfg.get("base_url")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    return OpenAI(api_key=api_key, base_url=base_url)

def safe_name(s: str, max_len: int = 80) -> str:
    """Make a filesystem-friendly short name."""
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:max_len] if len(s) > max_len else s

def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()
