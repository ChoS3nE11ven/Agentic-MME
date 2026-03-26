from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from common_utils import list_transformed_pngs, read_json
from ast_ops import PRIMITIVES

@dataclass
class VerifyResult:
    passed: bool
    hit_index: Optional[int] = None
    detail: Dict[str, Any] = None

def _is_declared_only(ev: Dict[str, Any]) -> bool:
    out = ev.get("output") or {}
    return bool(out.get("declared_only"))

def require_op(run_meta: Dict[str, Any], tool_use_list: List[Dict[str, Any]], op: str) -> VerifyResult:
    op = op.lower().strip()
    hits = []
    for ev in tool_use_list:
        if _is_declared_only(ev):
            continue
        tn = (ev.get("tool_name") or "").lower()
        arg_op = ((ev.get("arguments") or {}).get("op") or "").lower()
        if tn == op or arg_op == op:
            hits.append(ev)
    if not hits:
        return VerifyResult(False, None, {"reason": "op_not_found", "op": op})
    hit_index = int(hits[0].get("index", 0))
    return VerifyResult(True, hit_index, {"op": op, "matched_event": hits[0]})

def require_transformed_images(run_dir: Path, min_count: int = 1) -> VerifyResult:
    imgs = list_transformed_pngs(run_dir / "tool_images")
    ok = len(imgs) >= int(min_count)
    return VerifyResult(ok, 0 if ok else None, {"count": len(imgs), "min_count": int(min_count)})

def require_metrics_json(run_dir: Path, key: Optional[str] = None) -> VerifyResult:
    mp = run_dir / "tool_images" / "metrics.json"
    if not mp.exists():
        return VerifyResult(False, None, {"reason": "metrics.json_missing"})
    try:
        obj = read_json(mp)
    except Exception as e:
        return VerifyResult(False, None, {"reason": "metrics.json_invalid", "error": str(e)})
    if key and key not in obj:
        return VerifyResult(False, None, {"reason": "metrics.json_missing_key", "key": key})
    return VerifyResult(True, 0, {"metrics_keys": list(obj.keys())})

def auto_verifier_from_name(name: str) -> Optional[str]:
    low = (name or "").lower()
    for p in PRIMITIVES:
        if p in low:
            return f"require_op:{p}"
    return None

def dispatch_verifier(verifier: str, run_dir: Path, run_meta: Dict[str, Any], task_cfg: Dict[str, Any], checkpoint: Dict[str, Any], tool_use_list: List[Dict[str, Any]]) -> VerifyResult:
    verifier = (verifier or "").strip()
    if not verifier:
        return VerifyResult(True, None, {"note": "empty_verifier_treated_as_pass"})

    mapped = auto_verifier_from_name(verifier)
    if mapped and verifier.startswith("verify_") and verifier.endswith("_ast"):
        verifier = mapped

    if verifier.startswith("require_op:"):
        return require_op(run_meta, tool_use_list, verifier.split(":",1)[1])

    if verifier.startswith("require_images"):
        parts = verifier.split(":")
        n = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 1
        return require_transformed_images(run_dir, min_count=n)

    if verifier.startswith("require_metrics"):
        parts = verifier.split(":")
        key = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        return require_metrics_json(run_dir, key=key)

    fn = globals().get(verifier)
    if callable(fn):
        out = fn(run_dir=run_dir, run_meta=run_meta, task_cfg=task_cfg, checkpoint=checkpoint, tool_use_list=tool_use_list)
        if isinstance(out, VerifyResult):
            return out
        if isinstance(out, dict):
            return VerifyResult(bool(out.get("passed", False)), out.get("hit_index"), out)
        return VerifyResult(bool(out), None, {"note": "custom_fn_returned_nonstandard"})
    return VerifyResult(False, None, {"reason": "unknown_verifier", "verifier": verifier})

def example_custom_verifier(run_dir: Path, run_meta: Dict[str, Any], task_cfg: Dict[str, Any], checkpoint: Dict[str, Any], tool_use_list: List[Dict[str, Any]]) -> VerifyResult:
    return require_op(run_meta, tool_use_list, "crop")
