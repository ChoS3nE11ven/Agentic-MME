from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# Optional: OpenCV for advanced operations
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from common_utils import ensure_dir


def denormalize_bbox(bbox: List[float], width: int, height: int) -> List[int]:
    """Convert normalized bbox (0-1000) to absolute pixel coordinates."""
    abs_x1 = int(bbox[0] / 1000 * width)
    abs_y1 = int(bbox[1] / 1000 * height)
    abs_x2 = int(bbox[2] / 1000 * width)
    abs_y2 = int(bbox[3] / 1000 * height)
    
    # Ensure correct order
    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1
    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1
    
    # Clamp to boundaries
    abs_x1 = max(0, min(abs_x1, width))
    abs_x2 = max(0, min(abs_x2, width))
    abs_y1 = max(0, min(abs_y1, height))
    abs_y2 = max(0, min(abs_y2, height))
    
    return [abs_x1, abs_y1, abs_x2, abs_y2]


@dataclass
class AtomicState:
    """State manager for atomic tools with image index tracking."""
    orig_path: Path
    processed_dir: Path
    # Image array: index 0 is original, 1+ are processed images
    # images[i] = (path, label/description)
    images: List[tuple] = field(default_factory=list)
    step: int = 1  # Start from 1 so transformed_image_1.png corresponds to Image 1

    def __post_init__(self):
        ensure_dir(self.processed_dir)
        # Index 0 = original image
        self.images = [(self.orig_path, "original input image")]

    def get_image(self, index: int) -> Path:
        """Get image path by 0-based index."""
        if index < 0 or index >= len(self.images):
            raise ValueError(f"Invalid image_index {index}. Valid range: 0-{len(self.images)-1}")
        return self.images[index][0]

    def add_image(self, path: Path, label: str = "") -> int:
        """Add a new image and return its 0-based index."""
        self.images.append((path, label))
        return len(self.images) - 1

    def next_out(self) -> Path:
        """Generate next output path. transformed_image_N.png where N = step."""
        p = self.processed_dir / f"transformed_image_{self.step}.png"
        self.step += 1
        return p

    def get_image_list(self) -> List[Dict[str, Any]]:
        """Get list of all images with their indices for conversation tracking."""
        return [
            {"index": i, "path": str(p), "label": label}
            for i, (p, label) in enumerate(self.images)
        ]


def _open(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


# ============================================================================
# Image Tools with image_index and normalized bbox support
# ============================================================================

def tool_crop(state: AtomicState, image_index: int, bbox_2d: List[float], label: str = "", zoom_scale: float = 1.0) -> Dict[str, Any]:
    """Crop/zoom into a region of the specified image."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    w, h = img.size
    
    # Denormalize bbox from 0-1000 to absolute pixels
    x1, y1, x2, y2 = denormalize_bbox(bbox_2d, w, h)
    
    # Crop
    cropped = img.crop((x1, y1, x2, y2))
    
    # Apply zoom scale if > 1.0
    if zoom_scale > 1.0:
        new_w = int(cropped.width * zoom_scale)
        new_h = int(cropped.height * zoom_scale)
        cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
    
    out_path = state.next_out()
    cropped.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"crop from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "crop", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path),
        "bbox_normalized": bbox_2d,
        "bbox_absolute": [x1, y1, x2, y2],
        "zoom_scale": zoom_scale,
        "label": label,
        "size": list(cropped.size)
    }


def tool_rotate(state: AtomicState, image_index: int, angle: float, expand: bool = True, label: str = "") -> Dict[str, Any]:
    """Rotate the specified image."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    out = img.rotate(float(angle), expand=bool(expand))
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"rotate {angle}° from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "rotate", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path), 
        "angle": float(angle), 
        "expand": bool(expand), 
        "label": label,
        "size": list(out.size)
    }


def tool_flip(state: AtomicState, image_index: int, direction: str = "horizontal", label: str = "") -> Dict[str, Any]:
    """Flip/mirror the specified image.
    
    Args:
        state: AtomicState object
        image_index: Index of the image to flip
        direction: "horizontal" (left-right), "vertical" (top-bottom), or "both"
        label: Optional label for the output image
    
    Returns:
        Dictionary with operation results
    """
    img_path = state.get_image(image_index)
    img = _open(img_path)
    
    direction = direction.lower()
    if direction in ("horizontal", "h", "left_right", "lr"):
        # Horizontal flip (left-right mirror)
        out = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_desc = "horizontal"
    elif direction in ("vertical", "v", "top_bottom", "tb"):
        # Vertical flip (top-bottom mirror)
        out = img.transpose(Image.FLIP_TOP_BOTTOM)
        flip_desc = "vertical"
    elif direction in ("both", "b", "hv"):
        # Both directions (equivalent to 180° rotation)
        out = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        flip_desc = "both"
    else:
        # Default to horizontal if invalid direction
        out = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_desc = "horizontal"
    
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"flip {flip_desc} from image {image_index}")
    
    return {
        "ok": "true",
        "op": "flip",
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path),
        "direction": flip_desc,
        "label": label,
        "size": list(out.size)
    }


def tool_resize(state: AtomicState, image_index: int, width: Optional[int] = None, height: Optional[int] = None, scale: Optional[float] = None, label: str = "") -> Dict[str, Any]:
    """Resize the specified image."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    w, h = img.size
    
    if width is None or height is None:
        if scale is None:
            raise ValueError("resize requires (width, height) or scale")
        width = int(w * float(scale))
        height = int(h * float(scale))
    
    out = img.resize((int(width), int(height)), Image.LANCZOS)
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"resize from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "resize", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path), 
        "size": [int(width), int(height)], 
        "scale": float(scale) if scale else None,
        "label": label
    }


def tool_enhance(state: AtomicState, image_index: int, brightness: Optional[float] = None, contrast: Optional[float] = None, sharpness: Optional[float] = None, label: str = "") -> Dict[str, Any]:
    """Enhance brightness/contrast/sharpness of the specified image."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    
    if brightness is not None:
        img = ImageEnhance.Brightness(img).enhance(float(brightness))
    if contrast is not None:
        img = ImageEnhance.Contrast(img).enhance(float(contrast))
    if sharpness is not None:
        img = ImageEnhance.Sharpness(img).enhance(float(sharpness))
    
    out_path = state.next_out()
    img.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"enhance from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "enhance", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path), 
        "brightness": brightness, 
        "contrast": contrast, 
        "sharpness": sharpness,
        "label": label,
        "size": list(img.size)
    }


def tool_grayscale(state: AtomicState, image_index: int, label: str = "") -> Dict[str, Any]:
    """Convert the specified image to grayscale."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    out = img.convert("L").convert("RGB")
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"grayscale from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "grayscale", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path),
        "label": label,
        "size": list(out.size)
    }


def tool_autocontrast(state: AtomicState, image_index: int, cutoff: float = 0, label: str = "") -> Dict[str, Any]:
    """Apply automatic contrast adjustment."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    out = ImageOps.autocontrast(img, cutoff=float(cutoff))
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"autocontrast from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "autocontrast", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path), 
        "cutoff": cutoff,
        "label": label,
        "size": list(out.size)
    }


def tool_blur(state: AtomicState, image_index: int, radius: int = 2, label: str = "") -> Dict[str, Any]:
    """Apply Gaussian blur."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    out = img.filter(ImageFilter.GaussianBlur(radius=int(radius)))
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"blur from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "blur", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path), 
        "radius": radius,
        "label": label,
        "size": list(out.size)
    }


def tool_sharpen(state: AtomicState, image_index: int, label: str = "") -> Dict[str, Any]:
    """Apply sharpening filter."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    out = img.filter(ImageFilter.SHARPEN)
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"sharpen from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "sharpen", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path),
        "label": label,
        "size": list(out.size)
    }



def tool_denoise(state: AtomicState, image_index: int, strength: int = 10, label: str = "") -> Dict[str, Any]:
    """Remove noise from the image."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    
    if HAS_CV2:
        img_array = np.array(img)
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21)
        out = Image.fromarray(denoised)
    else:
        radius = max(1, strength // 3)
        out = img.filter(ImageFilter.MedianFilter(size=radius * 2 + 1))
    
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"denoise from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "denoise", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path), 
        "strength": strength,
        "label": label,
        "size": list(out.size)
    }


def tool_edge_detect(state: AtomicState, image_index: int, method: str = "canny", label: str = "") -> Dict[str, Any]:
    """Detect edges in the image."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    
    if method.lower() == "canny" and HAS_CV2:
        img_array = np.array(img.convert("L"))
        edges = cv2.Canny(img_array, 100, 200)
        out = Image.fromarray(edges).convert("RGB")
    elif method.lower() == "sobel" and HAS_CV2:
        img_array = np.array(img.convert("L"))
        sobelx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(np.clip(edges, 0, 255))
        out = Image.fromarray(edges).convert("RGB")
    else:
        out = img.filter(ImageFilter.FIND_EDGES).convert("RGB")
    
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"edge_detect from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "edge_detect", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path), 
        "method": method,
        "label": label,
        "size": list(out.size)
    }


def tool_invert(state: AtomicState, image_index: int, label: str = "") -> Dict[str, Any]:
    """Invert the colors of the image."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    out = ImageOps.invert(img)
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"invert from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "invert", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path),
        "label": label,
        "size": list(out.size)
    }


def tool_equalize(state: AtomicState, image_index: int, label: str = "") -> Dict[str, Any]:
    """Equalize the histogram of the image."""
    img_path = state.get_image(image_index)
    img = _open(img_path)
    out = ImageOps.equalize(img)
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"equalize from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "equalize", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path),
        "label": label,
        "size": list(out.size)
    }


def tool_threshold(state: AtomicState, image_index: int, value: int = 128, mode: str = "binary", label: str = "") -> Dict[str, Any]:
    """Apply threshold to convert image to binary."""
    img_path = state.get_image(image_index)
    img = _open(img_path).convert("L")
    
    if mode.lower() == "binary":
        out = img.point(lambda x: 255 if x > value else 0)
    elif mode.lower() == "binary_inv":
        out = img.point(lambda x: 0 if x > value else 255)
    elif mode.lower() == "trunc":
        out = img.point(lambda x: value if x > value else x)
    elif mode.lower() == "tozero":
        out = img.point(lambda x: x if x > value else 0)
    else:
        out = img.point(lambda x: 255 if x > value else 0)
    
    out = out.convert("RGB")
    out_path = state.next_out()
    out.save(out_path, "PNG")
    new_index = state.add_image(out_path, label or f"threshold from image {image_index}")
    
    return {
        "ok": "true", 
        "op": "threshold", 
        "source_image_index": image_index,
        "new_image_index": new_index,
        "output_path": str(out_path), 
        "value": value, 
        "mode": mode,
        "label": label,
        "size": list(out.size)
    }


# ============================================================================
# Tool Schema for OpenAI Function Calling
# ============================================================================

def build_atomic_tools_schema() -> list[dict]:
    """Build OpenAI tools schema with image_index and normalized bbox."""
    
    # Common image_index parameter
    image_index_param = {
        "type": "integer",
        "description": "Index of the image to operate on (0 = original, 1, 2... = processed images)",
        "minimum": 0
    }
    
    # Common label parameter
    label_param = {
        "type": "string",
        "description": "Description of what this operation is for"
    }
    
    return [
        {"type": "function", "function": {
            "name": "crop",
            "description": "Crop/zoom into a region of an image using normalized bounding box coordinates (0-1000 scale).",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Bounding box [x1, y1, x2, y2] in 0-1000 normalized coordinates"
                },
                "zoom_scale": {
                    "type": "number",
                    "description": "Magnification factor for the cropped region (default: 1.0). Higher values produce larger images.",
                    "minimum": 0.5,
                    "maximum": 5.0,
                    "default": 1.0
                },
                "label": label_param
            }, "required": ["image_index", "bbox_2d"]}
        }},
        {"type": "function", "function": {
            "name": "rotate",
            "description": "Rotate an image by angle degrees (positive = counterclockwise).",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "angle": {"type": "number", "description": "Rotation angle in degrees"},
                "expand": {"type": "boolean", "description": "If true, expand canvas to fit rotated image. Default: true"},
                "label": label_param
            }, "required": ["image_index", "angle"]}
        }},
        {"type": "function", "function": {
            "name": "flip",
            "description": "Flip/mirror an image horizontally, vertically, or both directions.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "direction": {
                    "type": "string",
                    "enum": ["horizontal", "vertical", "both"],
                    "description": "Flip direction: 'horizontal' (left-right mirror), 'vertical' (top-bottom mirror), or 'both'. Default: horizontal"
                },
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "resize",
            "description": "Resize an image to (width, height) OR by scale factor.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "width": {"type": "integer", "description": "Target width in pixels"},
                "height": {"type": "integer", "description": "Target height in pixels"},
                "scale": {"type": "number", "description": "Scale factor (e.g., 2.0 = double size, 0.5 = half)"},
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "enhance",
            "description": "Adjust brightness, contrast, and/or sharpness. Values: 1.0 = no change, >1.0 = increase, <1.0 = decrease.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "brightness": {"type": "number", "description": "Brightness multiplier (1.0 = no change)"},
                "contrast": {"type": "number", "description": "Contrast multiplier (1.0 = no change)"},
                "sharpness": {"type": "number", "description": "Sharpness multiplier (1.0 = no change)"},
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "grayscale",
            "description": "Convert an image to grayscale.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "autocontrast",
            "description": "Apply automatic contrast adjustment.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "cutoff": {"type": "number", "description": "Percentage of lightest/darkest pixels to ignore. Default: 0"},
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "blur",
            "description": "Apply Gaussian blur.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "radius": {"type": "integer", "description": "Blur radius. Default: 2"},
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "sharpen",
            "description": "Apply sharpening filter.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "denoise",
            "description": "Remove noise from an image.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "strength": {"type": "integer", "description": "Denoising strength (1-30). Default: 10"},
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "edge_detect",
            "description": "Detect edges in an image.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "method": {"type": "string", "enum": ["canny", "sobel", "simple"], "description": "Edge detection method. Default: canny"},
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "invert",
            "description": "Invert the colors of an image (create negative).",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "equalize",
            "description": "Equalize histogram for better contrast distribution.",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "label": label_param
            }, "required": ["image_index"]}
        }},
        {"type": "function", "function": {
            "name": "threshold",
            "description": "Apply threshold to convert image to binary (black/white).",
            "parameters": {"type": "object", "properties": {
                "image_index": image_index_param,
                "value": {"type": "integer", "description": "Threshold value (0-255). Default: 128"},
                "mode": {"type": "string", "enum": ["binary", "binary_inv", "trunc", "tozero"], "description": "Threshold mode. Default: binary"},
                "label": label_param
            }, "required": ["image_index"]}
        }},
    ]
