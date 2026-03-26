#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from common_utils import ensure_dir, read_json, write_json, image_to_data_url, make_openai_client
from dataset_utils import resolve_dataset_root, resolve_image_path

SYSTEM_PROMPT = "You are a helpful multimodal assistant. Return only the final answer."

def run_one(client: Any, task_json: Path, dataset_root: Path, images_dir: Optional[Path], out_dir: Path, model: str, temperature: float) -> Dict[str, Any]:
    task_cfg = read_json(task_json)
    img_path = resolve_image_path(task_json, task_cfg, dataset_root, images_dir)

    run_dir = ensure_dir(out_dir / task_json.stem)
    orig_copy = run_dir / "orig.png"
    Image.open(img_path).save(orig_copy, "PNG")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_to_data_url(orig_copy)}},
            {"type": "text", "text": (task_cfg.get("input") or {}).get("prompt","")},
        ]},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=2000)
    answer = (resp.choices[0].message.content or "").strip()
    (run_dir / "model_answer.txt").write_text(answer, encoding="utf-8")

    usage = {"api_calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if getattr(resp, "usage", None):
        usage["prompt_tokens"] = getattr(resp.usage, "prompt_tokens", 0) or 0
        usage["completion_tokens"] = getattr(resp.usage, "completion_tokens", 0) or 0
        usage["total_tokens"] = getattr(resp.usage, "total_tokens", 0) or 0

    run_meta = {
        "task_id": task_cfg.get("task_id",""),
        "task_file": str(task_json.resolve()),
        "mode": "no_tool",
        "driver": "direct",
        "model": model,
        "temperature": temperature,
        "usage": usage,
        "effective_tool_calls": 0,
        "paths": {"run_dir": str(run_dir), "orig": str(orig_copy), "processed_dir": ""},
    }
    write_json(run_dir / "run_meta.json", run_meta)
    return run_meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_json", type=str, default="")
    ap.add_argument("--task_dir", type=str, default="")
    ap.add_argument("--dataset_root", type=str, default="")
    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="runs_notool")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--base_url", type=str, default="")
    ap.add_argument("--api_config", type=str, default="")
    args = ap.parse_args()

    client = make_openai_client(api_key=args.api_key or None, base_url=args.base_url or None, api_config=Path(args.api_config) if args.api_config else None)

    tasks: List[Path] = []
    if args.task_json:
        tasks = [Path(args.task_json)]
    elif args.task_dir:
        tasks = sorted(Path(args.task_dir).glob("*.json"))
    else:
        raise ValueError("Provide --task_json or --task_dir")

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    images_dir = Path(args.images_dir) if args.images_dir else None
    out_dir = Path(args.out_dir)

    all_meta = []
    for t in tasks:
        try:
            ds = resolve_dataset_root(t, dataset_root)
            m = run_one(client, t, ds, images_dir, out_dir, args.model, args.temperature)
            all_meta.append(m)
            print(f"[OK] {t.name} -> {out_dir/t.stem}")
        except Exception as e:
            print(f"[ERR] {t}: {e}")
            traceback.print_exc()

    ensure_dir(out_dir)
    write_json(out_dir / "summary_runs.json", {"count": len(all_meta), "runs": all_meta})

if __name__ == "__main__":
    main()
