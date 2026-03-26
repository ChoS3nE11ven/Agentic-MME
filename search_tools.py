from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
SERPER_LENS_ENDPOINT = "https://google.serper.dev/lens"
IMGBB_UPLOAD_ENDPOINT = "https://api.imgbb.com/1/upload"

@dataclasses.dataclass
class SearchConfig:
    """Runtime configuration for search tools."""

    serper_api_key: str
    # Optional: needed only when you call Lens on a local image path.
    imgbb_api_key: Optional[str] = None
    # Optional: needed for fetch_webpage via Jina Reader
    jina_api_key: Optional[str] = None

    # Cache directory; if set, tool calls are recorded for replay.
    cache_dir: Optional[str] = None
    replay: bool = False

    request_timeout_s: int = 45
    default_gl: str = "us"
    default_hl: str = "en"

    @staticmethod
    def from_env(cache_dir_fallback: Optional[str] = None) -> "SearchConfig":
        serper_api_key = os.environ.get("SERPER_API_KEY", "").strip()
        if not serper_api_key:
            raise ValueError(
                "Missing SERPER_API_KEY. Set it as an environment variable (do NOT hardcode keys)."
            )

        imgbb_api_key = os.environ.get("IMGBB_API_KEY", "").strip() or None
        jina_api_key = os.environ.get("JINA_API_KEY", "").strip() or None

        cache_dir = os.environ.get("SEARCH_CACHE_DIR", "").strip() or cache_dir_fallback
        replay = os.environ.get("SEARCH_REPLAY", "0").strip() in {"1", "true", "True"}

        default_gl = os.environ.get("SERPER_GL", "us").strip() or "us"
        default_hl = os.environ.get("SERPER_HL", "en").strip() or "en"

        timeout_s_raw = os.environ.get("SEARCH_TIMEOUT_S", "").strip()
        timeout_s = int(timeout_s_raw) if timeout_s_raw.isdigit() else 45

        return SearchConfig(
            serper_api_key=serper_api_key,
            imgbb_api_key=imgbb_api_key,
            jina_api_key=jina_api_key,
            cache_dir=cache_dir,
            replay=replay,
            request_timeout_s=timeout_s,
            default_gl=default_gl,
            default_hl=default_hl,
        )


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_key(tool_name: str, payload: Dict[str, Any]) -> str:
    # Stable JSON so the same request always maps to the same cache file.
    stable = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{tool_name}_{_sha256_hex(stable)[:16]}"


def _get_next_cache_index(cache_dir: str, task_id: str, tool_prefix: str) -> int:
    """Get the next available index for cache file naming."""
    task_cache_dir = os.path.join(cache_dir, task_id)
    if not os.path.exists(task_cache_dir):
        return 1
    
    existing = []
    for f in os.listdir(task_cache_dir):
        if f.startswith(tool_prefix) and f.endswith(".json"):
            # Extract index from filename like "serper_search_1.json"
            try:
                idx = int(f[len(tool_prefix):-5])  # Remove prefix and ".json"
                existing.append(idx)
            except ValueError:
                pass
    
    return max(existing, default=0) + 1


def _find_existing_cache(cache_dir: str, task_id: str, tool_prefix: str, payload: Dict[str, Any]) -> Optional[str]:
    """Find existing cache file that matches the payload."""
    task_cache_dir = os.path.join(cache_dir, task_id)
    if not os.path.exists(task_cache_dir):
        return None
    
    for f in os.listdir(task_cache_dir):
        if f.startswith(tool_prefix) and f.endswith(".json"):
            cache_path = os.path.join(task_cache_dir, f)
            cached = _read_cache(cache_path)
            if cached and cached.get("request") == payload:
                return cache_path
    
    return None


def _ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def _read_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_cache(cache_path: str, obj: Dict[str, Any]) -> None:
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)


def _post_json_with_retry(url: str, api_key: str, payload: Dict[str, Any], timeout_s: int, max_retries: int = 3) -> Dict[str, Any]:
    """POST JSON with automatic retry on timeout/5xx errors."""
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    
    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if r.status_code >= 500:
                # Server error, retry
                last_error = RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:500]}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                    print(f"[Search] Server error {r.status_code}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                raise last_error
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:500]}")
            return r.json()
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"[Search] Timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Request timeout after {max_retries} attempts: {url}")
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"[Search] Connection error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Connection error after {max_retries} attempts: {e}")
    
    raise last_error or RuntimeError(f"Failed after {max_retries} attempts")


def _post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    """POST JSON with automatic retry."""
    return _post_json_with_retry(url, api_key, payload, timeout_s, max_retries=3)


def _compress_image_for_upload(image_path: str, max_size_bytes: int = 30 * 1024 * 1024) -> bytes:
    """Compress image if it exceeds max_size_bytes (default 30MB, leaving margin for ImgBB's 32MB limit).
    
    Strategy (prioritize quality reduction over resizing):
    1. First try original file
    2. If too large, reduce JPEG quality first (keep original size)
    3. Only resize if quality reduction alone isn't enough
    
    Returns: image bytes ready for base64 encoding
    """
    from PIL import Image
    import io
    
    # Read original file
    with open(image_path, "rb") as f:
        original_bytes = f.read()
    
    # If small enough, return as-is
    if len(original_bytes) <= max_size_bytes:
        return original_bytes
    
    # Need to compress
    img = Image.open(image_path)
    
    # Convert to RGB if necessary (for JPEG compression)
    if img.mode in ('RGBA', 'P', 'LA'):
        # Create white background for transparent images
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    original_size = img.size
    
    # Strategy: prioritize quality reduction, only resize as last resort
    # First: try quality reduction at full size
    quality_levels = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]
    
    for quality in quality_levels:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        compressed_bytes = buffer.getvalue()
        
        if len(compressed_bytes) <= max_size_bytes:
            print(f"[ImgBB] Compressed image: {len(original_bytes)/1024/1024:.1f}MB -> {len(compressed_bytes)/1024/1024:.1f}MB "
                  f"(quality={quality}, size=original)")
            return compressed_bytes
    
    # If quality alone isn't enough, try mild resizing + quality
    scale_factors = [0.9, 0.8, 0.7]  # Only mild resizing
    
    for scale in scale_factors:
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        for quality in [50, 40, 30]:
            buffer = io.BytesIO()
            resized.save(buffer, format='JPEG', quality=quality, optimize=True)
            compressed_bytes = buffer.getvalue()
            
            if len(compressed_bytes) <= max_size_bytes:
                print(f"[ImgBB] Compressed image: {len(original_bytes)/1024/1024:.1f}MB -> {len(compressed_bytes)/1024/1024:.1f}MB "
                      f"(scale={scale}, quality={quality})")
                return compressed_bytes
    
    # Last resort: more aggressive but still reasonable
    final_scale = 0.6
    new_size = (int(original_size[0] * final_scale), int(original_size[1] * final_scale))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format='JPEG', quality=25, optimize=True)
    compressed_bytes = buffer.getvalue()
    
    print(f"[ImgBB] Compressed image: {len(original_bytes)/1024/1024:.1f}MB -> {len(compressed_bytes)/1024/1024:.1f}MB "
          f"(scale={final_scale}, quality=25)")
    return compressed_bytes


def imgbb_upload_image(image_path: str, api_key: str, timeout_s: int = 45) -> Dict[str, Any]:
    """Upload an image file to ImgBB and return the response.

    Notes:
    - ImgBB expects base64 image content in form field `image`.
    - The returned JSON contains a `data.url` field that is publicly accessible.
    - Images larger than 30MB are automatically compressed before upload.
    """

    # Compress if needed (ImgBB has 32MB limit, we use 30MB to be safe)
    image_bytes = _compress_image_for_upload(image_path, max_size_bytes=30 * 1024 * 1024)
    b64 = base64.b64encode(image_bytes).decode("ascii")

    # Per ImgBB API, key is a query param.
    r = requests.post(
        IMGBB_UPLOAD_ENDPOINT,
        params={"key": api_key},
        data={"image": b64},
        timeout=timeout_s,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"ImgBB upload failed (HTTP {r.status_code}): {r.text[:500]}")
    return r.json()


def ensure_public_image_url(
    *,
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    cfg: SearchConfig,
) -> str:
    """Return a public URL for the image.

    - If `image_url` is already an http(s) URL, returns it.
    - Else, uploads `image_path` using ImgBB (requires IMGBB_API_KEY).
    """

    if image_url:
        if image_url.startswith("http://") or image_url.startswith("https://"):
            return image_url
        raise ValueError(f"image_url must be http(s), got: {image_url[:50]}")

    if not image_path:
        raise ValueError("Provide either image_url or image_path")

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    if not cfg.imgbb_api_key:
        raise ValueError(
            "Lens search needs a public image URL. Provide IMGBB_API_KEY (or change uploader) "
            "so local images/crops can be uploaded."
        )

    uploaded = imgbb_upload_image(image_path, cfg.imgbb_api_key, timeout_s=cfg.request_timeout_s)
    try:
        return uploaded["data"]["url"]
    except Exception as e:
        raise RuntimeError(f"ImgBB response missing data.url: {uploaded}") from e


def _compact_serper_search(resp: Dict[str, Any], max_items: int = 5) -> str:
    """Create a compact text block for the LLM context."""

    lines: List[str] = []

    kg = resp.get("knowledgeGraph")
    if isinstance(kg, dict):
        title = kg.get("title")
        desc = kg.get("description")
        website = kg.get("website")
        if title:
            lines.append(f"[KnowledgeGraph] {title}")
        if desc:
            lines.append(f"  - {desc}")
        if website:
            lines.append(f"  - {website}")
        lines.append("")

    organic = resp.get("organic") or []
    if isinstance(organic, list):
        lines.append("[Organic]")
        for i, item in enumerate(organic[:max_items], start=1):
            if not isinstance(item, dict):
                continue
            t = (item.get("title") or "").strip()
            link = (item.get("link") or "").strip()
            snip = (item.get("snippet") or "").strip()
            date = (item.get("date") or "").strip()
            if not (t or link or snip):
                continue
            head = f"{i}. {t}" if t else f"{i}."
            if date:
                head += f" ({date})"
            lines.append(head)
            if link:
                lines.append(f"   {link}")
            if snip:
                lines.append(f"   {snip}")
        lines.append("")

    paa = resp.get("peopleAlsoAsk") or []
    if isinstance(paa, list) and paa:
        lines.append("[PeopleAlsoAsk]")
        for item in paa[:5]:
            if not isinstance(item, dict):
                continue
            q = (item.get("question") or "").strip()
            snip = (item.get("snippet") or "").strip()
            link = (item.get("link") or "").strip()
            if q:
                lines.append(f"- Q: {q}")
                if snip:
                    lines.append(f"  A: {snip}")
                if link:
                    lines.append(f"  {link}")
        lines.append("")

    return "\n".join(lines).strip()


def _compact_serper_lens(resp: Dict[str, Any], max_items: int = 5) -> str:
    lines: List[str] = []

    organic = resp.get("organic") or []
    if isinstance(organic, list):
        lines.append("[Lens Organic]")
        for i, item in enumerate(organic[:max_items], start=1):
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            source = (item.get("source") or "").strip()
            link = (item.get("link") or "").strip()
            thumbnail_url = (item.get("thumbnailUrl") or "").strip()
            if not (title or link):
                continue
            head = f"{i}. {title}" if title else f"{i}."
            if source:
                head += f" [{source}]"
            lines.append(head)
            if link:
                lines.append(f"   {link}")
            if thumbnail_url:
                lines.append(f"   thumbnailUrl: {thumbnail_url}")
        lines.append("")

    return "\n".join(lines).strip()


def google_search(
    *,
    query: str,
    cfg: SearchConfig,
    gl: Optional[str] = None,
    hl: Optional[str] = None,
    page: int = 1,
    autocorrect: bool = True,
    search_type: str = "search",
    task_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Serper Google Search."""

    payload = {
        "q": query,
        "gl": gl or cfg.default_gl,
        "hl": hl or cfg.default_hl,
        "autocorrect": bool(autocorrect),
        "page": int(page),
        "type": search_type,
    }

    cache_dir = cfg.cache_dir
    cache_path = None
    
    if cache_dir and task_id:
        task_cache_dir = os.path.join(cache_dir, task_id)
        _ensure_dir(task_cache_dir)
        
        # Check if we already have a cache for this exact request
        existing_cache = _find_existing_cache(cache_dir, task_id, "serper_search_", payload)
        
        if cfg.replay:
            if existing_cache is None:
                raise FileNotFoundError(f"Replay mode: missing cache for task {task_id}")
            cached = _read_cache(existing_cache)
            resp = cached["response"]
            return resp, _compact_serper_search(resp)

        if existing_cache:
            cached = _read_cache(existing_cache)
            if cached is not None:
                resp = cached["response"]
                return resp, _compact_serper_search(resp)
        
        # Create new cache file with next index
        next_idx = _get_next_cache_index(cache_dir, task_id, "serper_search_")
        cache_path = os.path.join(task_cache_dir, f"serper_search_{next_idx}.json")

    resp = _post_json(
        SERPER_SEARCH_ENDPOINT,
        api_key=cfg.serper_api_key,
        payload=payload,
        timeout_s=cfg.request_timeout_s,
    )

    if cache_path:
        _write_cache(
            cache_path,
            {
                "tool": "serper_search",
                "timestamp": time.time(),
                "request": payload,
                "response": resp,
            },
        )

    return resp, _compact_serper_search(resp)


def google_lens_search(
    *,
    cfg: SearchConfig,
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    page: int = 1,
    num: int = 10,
    task_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str, str]:
    """Serper Google Lens.

    Returns: (raw_response, compact_context, final_image_url)
    """

    final_url = ensure_public_image_url(image_url=image_url, image_path=image_path, cfg=cfg)

    payload = {
        "url": final_url,
        "page": int(page),
        "num": int(num),
    }

    cache_dir = cfg.cache_dir
    cache_path = None
    
    if cache_dir and task_id:
        task_cache_dir = os.path.join(cache_dir, task_id)
        _ensure_dir(task_cache_dir)
        
        # Check if we already have a cache for this exact request
        existing_cache = _find_existing_cache(cache_dir, task_id, "serper_lens_", payload)
        
        if cfg.replay:
            if existing_cache is None:
                raise FileNotFoundError(f"Replay mode: missing cache for task {task_id}")
            cached = _read_cache(existing_cache)
            resp = cached["response"]
            return resp, _compact_serper_lens(resp), final_url

        if existing_cache:
            cached = _read_cache(existing_cache)
            if cached is not None:
                resp = cached["response"]
                return resp, _compact_serper_lens(resp), final_url
        
        # Create new cache file with next index
        next_idx = _get_next_cache_index(cache_dir, task_id, "serper_lens_")
        cache_path = os.path.join(task_cache_dir, f"serper_lens_{next_idx}.json")

    resp = _post_json(
        SERPER_LENS_ENDPOINT,
        api_key=cfg.serper_api_key,
        payload=payload,
        timeout_s=cfg.request_timeout_s,
    )

    if cache_path:
        _write_cache(
            cache_path,
            {
                "tool": "serper_lens",
                "timestamp": time.time(),
                "request": payload,
                "response": resp,
            },
        )

    return resp, _compact_serper_lens(resp), final_url


def jina_read(url: str, api_key: Optional[str] = None, timeout_s: int = 30, max_retries: int = 3) -> str:
    """Optional: fetch a web page via Jina AI Reader (textified).

    This is useful when you want the model to cite/quote from specific pages
    after search.

    We do not cache this here because some users prefer caching only search JSON.
    You can wrap it with your own caching if needed.
    """

    # Jina Reader format: https://r.jina.ai/http(s)://...
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"url must be http(s), got: {url}")
    reader_url = "https://r.jina.ai/" + url
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.get(reader_url, headers=headers, timeout=timeout_s)
            if r.status_code >= 500:
                last_error = RuntimeError(f"Jina Reader failed (HTTP {r.status_code}): {r.text[:300]}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"[Jina] Server error {r.status_code}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                raise last_error
            if r.status_code >= 400:
                raise RuntimeError(f"Jina Reader failed (HTTP {r.status_code}): {r.text[:300]}")
            return r.text
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"[Jina] Network error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Jina Reader failed after {max_retries} attempts: {e}")
    
    raise last_error or RuntimeError(f"Jina Reader failed after {max_retries} attempts")




# --------------------------
# Convenience wrapper API (used by runners)
# --------------------------

def load_search_config(path: Optional[str] = None) -> SearchConfig:
    """Load SearchConfig from a JSON file or environment variables.

    JSON schema (example): search_config_example.json
    """
    if path:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"search_config not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("search_config JSON must be an object")

        serper_api_key = str(data.get("serper_api_key") or "").strip()
        if not serper_api_key:
            raise ValueError("search_config missing required field: serper_api_key")

        imgbb_api_key = (str(data.get("imgbb_api_key")).strip() if data.get("imgbb_api_key") else None) or None
        jina_api_key = (str(data.get("jina_api_key")).strip() if data.get("jina_api_key") else None) or None
        cache_dir = (str(data.get("cache_dir")).strip() if data.get("cache_dir") else None) or None
        replay = bool(data.get("replay", False))
        timeout_s = int(data.get("request_timeout_s", 45) or 45)
        default_gl = str(data.get("default_gl", "us") or "us").strip() or "us"
        default_hl = str(data.get("default_hl", "en") or "en").strip() or "en"

        return SearchConfig(
            serper_api_key=serper_api_key,
            imgbb_api_key=imgbb_api_key,
            jina_api_key=jina_api_key,
            cache_dir=cache_dir,
            replay=replay,
            request_timeout_s=timeout_s,
            default_gl=default_gl,
            default_hl=default_hl,
        )

    # Fallback to environment variables
    return SearchConfig.from_env()


def _prune_serper_search_response(resp: Dict[str, Any], max_items: int = 5) -> Dict[str, Any]:
    """Keep only the most useful parts of Serper /search response to reduce token size."""
    out: Dict[str, Any] = {}
    for k in ["searchParameters", "knowledgeGraph", "answerBox"]:
        if k in resp:
            out[k] = resp.get(k)
    organic = resp.get("organic")
    if isinstance(organic, list):
        out["organic"] = organic[:max_items]
    images = resp.get("images")
    if isinstance(images, list):
        out["images"] = images[:max_items]
    news = resp.get("news")
    if isinstance(news, list):
        out["news"] = news[:max_items]
    paa = resp.get("peopleAlsoAsk")
    if isinstance(paa, list):
        out["peopleAlsoAsk"] = paa[:max_items]
    related = resp.get("relatedSearches")
    if isinstance(related, list):
        out["relatedSearches"] = related[:max_items]
    return out


def _prune_serper_lens_response(resp: Dict[str, Any], max_items: int = 5) -> Dict[str, Any]:
    """Keep only the most useful parts of Serper /lens response to reduce token size."""
    out: Dict[str, Any] = {}
    for k in ["searchParameters", "visualMatches", "knowledgeGraph"]:
        if k in resp:
            out[k] = resp.get(k)
    organic = resp.get("organic")
    if isinstance(organic, list):
        out["organic"] = organic[:max_items]
    return out


def fetch_webpage(*, url: str, max_chars: int = 12000, timeout_s: int = 45, jina_api_key: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a webpage and convert it to clean-ish text via Jina Reader.

    Returns a dict with a truncated 'text' field to keep context compact.
    """
    text = jina_read(url=url, api_key=jina_api_key, timeout_s=timeout_s)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return {"url": url, "text": text}


def download_image_from_url(
    url: str,
    save_dir: str,
    filename: Optional[str] = None,
    timeout_s: int = 30,
    max_retries: int = 3,
    max_size_kb: int = 500,  # Compress to max 500KB to save costs
) -> Dict[str, Any]:
    """Download an image from URL, compress it, and save locally.
    
    Args:
        url: The URL of the image to download
        save_dir: Directory to save the downloaded image
        filename: Optional custom filename (auto-generated if not provided)
        timeout_s: Request timeout in seconds
        max_retries: Number of retry attempts
        max_size_kb: Maximum file size in KB after compression (default 500KB)
    
    Returns: {"ok": True/False, "path": saved_path, "filename": filename, "error": error_msg}
    """
    import mimetypes
    from PIL import Image
    import io
    
    if not url or not url.strip():
        return {"ok": False, "error": "Empty URL provided"}
    
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "error": f"Invalid URL (must be http/https): {url[:100]}"}
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                timeout=timeout_s, 
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
                allow_redirects=True,
            )
            response.raise_for_status()
            
            image_data = response.content
            
            # Validate it's actually an image
            if len(image_data) < 100:
                return {"ok": False, "error": "Downloaded content too small to be a valid image"}
            
            # Try to open as image to validate
            try:
                img = Image.open(io.BytesIO(image_data))
                img.load()  # Force load to validate
            except Exception as e:
                return {"ok": False, "error": f"Invalid image data: {str(e)[:100]}"}
            
            # Generate filename with standard naming: downloaded_image_N.png
            if not filename:
                # Find next available index
                save_path_dir = pathlib.Path(save_dir)
                save_path_dir.mkdir(parents=True, exist_ok=True)
                existing = [f for f in save_path_dir.iterdir() if f.name.startswith("downloaded_image_") and f.suffix == ".png"]
                indices = []
                for f in existing:
                    try:
                        idx = int(f.stem.replace("downloaded_image_", ""))
                        indices.append(idx)
                    except ValueError:
                        pass
                next_idx = max(indices, default=0) + 1
                filename = f"downloaded_image_{next_idx}.png"
            elif not filename.endswith(".png"):
                filename = filename.rsplit(".", 1)[0] + ".png"
            
            # Compress image to save costs
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (max 1024px on longest side for downloaded images)
            max_dim = 1024
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Compress with quality reduction
            max_bytes = max_size_kb * 1024
            quality = 85
            buffer = io.BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            
            # If still too large, convert to JPEG with quality reduction
            if buffer.tell() > max_bytes:
                for q in [80, 70, 60, 50, 40]:
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=q, optimize=True)
                    if buffer.tell() <= max_bytes:
                        quality = q
                        filename = filename.rsplit(".", 1)[0] + ".jpg"
                        break
            
            compressed_data = buffer.getvalue()
            
            # Save the image
            save_path = pathlib.Path(save_dir) / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(compressed_data)
            
            original_kb = len(image_data) / 1024
            final_kb = len(compressed_data) / 1024
            print(f"[download_image] Saved: {filename} ({final_kb:.1f}KB, compressed from {original_kb:.1f}KB)")
            
            return {
                "ok": True,
                "path": str(save_path),
                "filename": filename,
                "size_bytes": len(compressed_data),
                "original_size_bytes": len(image_data),
            }
            
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP error {e.response.status_code}: {str(e)[:100]}"
            if e.response.status_code in (403, 404, 410):
                return {"ok": False, "error": last_error}
        except requests.exceptions.Timeout:
            last_error = "Request timed out"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {str(e)[:100]}"
        except requests.exceptions.RequestException as e:
            last_error = f"Request failed: {str(e)[:100]}"
        except Exception as e:
            return {"ok": False, "error": f"Error processing image: {str(e)[:200]}"}
        
        if attempt < max_retries - 1:
            wait_time = 2 * (attempt + 1)
            print(f"[download_image] Retry {attempt + 1}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
    
    return {"ok": False, "error": f"Download failed after {max_retries} attempts: {last_error}"}


class SearchTools:
    """Thin OO wrapper so runners can hold cfg + cache_dir easily."""

    def __init__(self, cfg: SearchConfig, task_id: Optional[str] = None):
        self.cfg = cfg
        self.task_id = task_id

    def google_search(
        self,
        *,
        query: str,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        page: int = 1,
        search_type: str = "search",
        autocorrect: bool = True,
        max_items: int = 5,
    ) -> Dict[str, Any]:
        raw, context = google_search(
            query=query,
            cfg=self.cfg,
            gl=gl,
            hl=hl,
            page=page,
            autocorrect=autocorrect,
            search_type=search_type,
            task_id=self.task_id,
        )
        return {
            "tool": "google_search",
            "query": query,
            "gl": gl or self.cfg.default_gl,
            "hl": hl or self.cfg.default_hl,
            "page": int(page),
            "type": search_type,
            "context": context,
            "raw": _prune_serper_search_response(raw, max_items=max_items),
        }

    def google_lens_search(
        self,
        *,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        page: int = 1,
        num: int = 5,
        max_items: int = 5,
    ) -> Dict[str, Any]:
        # Back-compat: callers may pass a local path via image_url
        if image_url and not (image_url.startswith("http://") or image_url.startswith("https://")):
            image_path = image_url
            image_url = None

        raw, context, final_url = google_lens_search(
            cfg=self.cfg,
            image_url=image_url,
            image_path=image_path,
            page=page,
            num=num,
            task_id=self.task_id,
        )
        return {
            "tool": "google_lens_search",
            "page": int(page),
            "num": int(num),
            "final_image_url": final_url,
            "context": context,
            "raw": _prune_serper_lens_response(raw, max_items=max_items),
        }

    def fetch_webpage(self, *, url: str, max_chars: int = 12000) -> Dict[str, Any]:
        return fetch_webpage(url=url, max_chars=max_chars, timeout_s=self.cfg.request_timeout_s, jina_api_key=self.cfg.jina_api_key)
