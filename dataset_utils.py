from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

def extract_numeric_id_from_filename(p: Path) -> Optional[int]:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else None

def pad_id(n: int, width: int = 4) -> str:
    return str(n).zfill(width)

def resolve_dataset_root(task_json: Path, dataset_root: Optional[Path]) -> Path:
    if dataset_root is not None:
        return dataset_root
    return task_json.parent.parent

def resolve_image_path(task_json: Path, task_cfg: Dict[str, Any], dataset_root: Path, images_dir: Optional[Path]) -> Union[Path, List[Path]]:
    """
    Resolve image path(s) based on JSON filename.
    
    Matching logic:
    - JSON: 0009.json → Image: image_0009.png (single image)
    - JSON: 0162.json → Images: image_0162_1.png, image_0162_2.png, ... (multiple images)
    
    Returns:
        - Single Path if one image
        - List[Path] if multiple images
    """
    # Extract numeric ID from JSON filename
    num = extract_numeric_id_from_filename(task_json)
    if num is None:
        raise FileNotFoundError(f"Cannot infer numeric id from {task_json.name}")
    
    padded_id = pad_id(num, 4)  # e.g., "0009"
    
    # Search directories (in priority order)
    search_dirs = []
    if images_dir is not None:
        search_dirs.append(images_dir)
    search_dirs.extend([dataset_root / "image", dataset_root / "images"])
    
    # First, try to find multiple images: image_XXXX_1.png, image_XXXX_2.png, ...
    multi_images = []
    for search_dir in search_dirs:
        if search_dir is None or not search_dir.exists():
            continue
        
        # Look for image_XXXX_N.png pattern
        idx = 1
        while True:
            for ext in [".png", ".jpg", ".jpeg"]:
                pattern = f"image_{padded_id}_{idx}{ext}"
                p = (search_dir / pattern).resolve()
                if p.exists():
                    multi_images.append(p)
                    break
            else:
                # No image found for this index, stop searching
                break
            idx += 1
        
        # If we found multiple images in this directory, return them
        if multi_images:
            return multi_images
    
    # If no multiple images found, try single image: image_XXXX.png
    for search_dir in search_dirs:
        if search_dir is None or not search_dir.exists():
            continue
        
        for ext in [".png", ".jpg", ".jpeg"]:
            pattern = f"image_{padded_id}{ext}"
            p = (search_dir / pattern).resolve()
            if p.exists():
                return p
    
    raise FileNotFoundError(f"Cannot resolve image for {task_json.name}: tried image_{padded_id}.png and image_{padded_id}_N.png in {search_dirs}")


def _resolve_single_image(rel: str, task_json: Path, dataset_root: Path, images_dir: Optional[Path]) -> Path:
    """Resolve a single image path."""
    # Try relative to dataset_root
    p = (dataset_root / rel).resolve()
    if p.exists():
        return p
    
    # Try absolute path
    abs_p = Path(rel)
    if abs_p.is_absolute() and abs_p.exists():
        return abs_p
    
    # Try just the filename in images_dir
    fname = Path(rel).name
    if images_dir is not None:
        p = (images_dir / fname).resolve()
        if p.exists():
            return p
    
    # Try in dataset_root/image (new format)
    p = (dataset_root / "image" / fname).resolve()
    if p.exists():
        return p
    
    # Try in dataset_root/images (old format)
    p = (dataset_root / "images" / fname).resolve()
    if p.exists():
        return p
    
    raise FileNotFoundError(f"Cannot resolve image: {rel}")
