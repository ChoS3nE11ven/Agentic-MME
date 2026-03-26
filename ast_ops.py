from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_IMG_RE = re.compile(r"transformed_image_(\d+)\.png", re.IGNORECASE)
_ANY_IMG_RE = re.compile(r"\.(png|jpg|jpeg|bmp|gif|tiff?)$", re.IGNORECASE)

PRIMITIVES = ("crop", "rotate", "resize", "enhance", "flip")


@dataclass
class OpEvent:
    op: str
    lineno: int
    col: int
    detail: Dict[str, Any]


@dataclass
class SaveEvent:
    image_index: int
    lineno: int
    col: int
    filename: str
    op_guess: str
    is_standard_name: bool = True


def _attr_chain(n: ast.AST) -> str:
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    return ".".join(reversed(parts))


# ---------------------------
# Static loop length inference
# ---------------------------
def _eval_const_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval_const_int(node.operand)
        return -v if v is not None else None
    return None


def _range_len(call: ast.Call) -> Optional[int]:
    # range(stop) / range(start, stop) / range(start, stop, step)
    if not (isinstance(call.func, ast.Name) and call.func.id == "range"):
        return None
    args = call.args
    if not args:
        return None
    vals = [_eval_const_int(a) for a in args]
    if any(v is None for v in vals):
        return None

    if len(vals) == 1:
        start, stop, step = 0, vals[0], 1
    elif len(vals) == 2:
        start, stop = vals
        step = 1
    else:
        start, stop, step = vals[:3]

    if step == 0:
        return None
    if (step > 0 and start >= stop) or (step < 0 and start <= stop):
        return 0
    if step > 0:
        return (stop - start + step - 1) // step
    else:
        # step is negative
        return (stop - start + step + 1) // step


def _static_iter_len(node: ast.AST, var_len: Dict[str, int]) -> Optional[int]:
    # literals
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts)
    if isinstance(node, ast.Dict):
        return len(node.keys)

    # name -> previously known length
    if isinstance(node, ast.Name):
        return var_len.get(node.id)

    # call patterns
    if isinstance(node, ast.Call):
        # enumerate(x)
        if isinstance(node.func, ast.Name) and node.func.id == "enumerate" and node.args:
            return _static_iter_len(node.args[0], var_len)

        # range(...)
        rl = _range_len(node)
        if rl is not None:
            return rl

        # dict.items()/keys()/values()
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("items", "keys", "values"):
            base = node.func.value
            if isinstance(base, ast.Dict):
                return len(base.keys)
            if isinstance(base, ast.Name):
                return var_len.get(base.id)

    return None


# ---------------------------
# f-string pattern improvement
# ---------------------------
def _joinedstr_to_pattern(node: ast.JoinedStr) -> str:
    """
    f"transformed_image_{i}.png" -> "transformed_image_{i}.png"
    unknown formatted expr -> "{expr}"
    """
    parts: List[str] = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            parts.append(str(v.value))
        elif isinstance(v, ast.FormattedValue):
            if isinstance(v.value, ast.Name):
                parts.append("{" + v.value.id + "}")
            else:
                parts.append("{expr}")
    return "".join(parts)


def _extract_filename_pattern(node: ast.AST, var_to_filename: Optional[Dict[str, str]] = None) -> str:
    """
    Extract filename pattern from various path construction methods.
    Returns:
      - exact name if statically known
      - a pattern like transformed_image_{i}.png if f-string variable unknown
    """
    var_to_filename = var_to_filename or {}

    # Name resolution
    if isinstance(node, ast.Name):
        return var_to_filename.get(node.id, "")

    # Direct string constant
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    # os.path.join(...) or similar call
    if isinstance(node, ast.Call):
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if _IMG_RE.search(arg.value) or _ANY_IMG_RE.search(arg.value):
                    return arg.value
            elif isinstance(arg, ast.JoinedStr):
                result = _joinedstr_to_pattern(arg)
                if _IMG_RE.search(result) or _ANY_IMG_RE.search(result):
                    return result
            elif isinstance(arg, ast.Name):
                resolved = var_to_filename.get(arg.id, "")
                if resolved and (_IMG_RE.search(resolved) or _ANY_IMG_RE.search(resolved)):
                    return resolved

        for kw in getattr(node, "keywords", []):
            v = kw.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                if _IMG_RE.search(v.value) or _ANY_IMG_RE.search(v.value):
                    return v.value
            if isinstance(v, ast.Name):
                resolved = var_to_filename.get(v.id, "")
                if resolved and (_IMG_RE.search(resolved) or _ANY_IMG_RE.search(resolved)):
                    return resolved

    # Binary operations (e.g., Path / "filename")
    if isinstance(node, ast.BinOp):
        left_str = _extract_filename_pattern(node.left, var_to_filename)
        if left_str and (_IMG_RE.search(left_str) or _ANY_IMG_RE.search(left_str)):
            return left_str
        right_str = _extract_filename_pattern(node.right, var_to_filename)
        if right_str and (_IMG_RE.search(right_str) or _ANY_IMG_RE.search(right_str)):
            return right_str

    # f-strings
    if isinstance(node, ast.JoinedStr):
        result = _joinedstr_to_pattern(node)
        if _IMG_RE.search(result) or _ANY_IMG_RE.search(result):
            return result

    return ""


# ---------------------------
# Main infer + expansion + full tool events
# ---------------------------
def infer_ops_and_saves(code: str) -> Tuple[List[OpEvent], List[SaveEvent]]:
    """
    - Expand ops/saves inside statically-resolvable for-loops so UI can show each occurrence.
    - Safety limit: cap expansion at MAX_EXPAND.
    - Heuristic: if filename is transformed_image_{i}.png and we can prove i starts from a constant
      and increments by a constant step inside the loop, expand save filenames into concrete indices.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], []

    ops: List[OpEvent] = []
    saves_raw: List[Tuple[int, int, int, str, bool]] = []

    var_to_filename: Dict[str, str] = {}
    var_len: Dict[str, int] = {}

    # track constant int assignments to support i=2 heuristic
    var_const_int: Dict[str, int] = {}

    has_crop = False
    loop_mult_stack: List[int] = [1]
    # stack to know if we're in a specific for node (for heuristics)
    for_stack: List[ast.For] = []

    MAX_EXPAND = 50

    def _append_ops_n_times(evt: OpEvent, n: int) -> None:
        if n <= 1:
            ops.append(evt)
            return
        if n > MAX_EXPAND:
            # do not expand: keep a marker
            ops.append(
                OpEvent(
                    op=evt.op,
                    lineno=evt.lineno,
                    col=evt.col,
                    detail={**evt.detail, "loop_count": n, "expanded": False},
                )
            )
            return
        for _ in range(n):
            ops.append(OpEvent(op=evt.op, lineno=evt.lineno, col=evt.col, detail=dict(evt.detail)))

    def _append_save_raw_n_times(item: Tuple[int, int, int, str, bool], n: int) -> None:
        if n <= 1:
            saves_raw.append(item)
            return
        if n > MAX_EXPAND:
            saves_raw.append(item)
            return
        for _ in range(n):
            saves_raw.append(item)

    def _loop_inc_vars(for_node: ast.For) -> Dict[str, int]:
        """
        Scan for body, find i += 1 / i -= 1 (constant step only).
        Return {var: step}
        """
        inc: Dict[str, int] = {}
        for stmt in for_node.body:
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                step = _eval_const_int(stmt.value)
                if step is None:
                    continue
                if isinstance(stmt.op, ast.Add):
                    inc[stmt.target.id] = inc.get(stmt.target.id, 0) + step
                elif isinstance(stmt.op, ast.Sub):
                    inc[stmt.target.id] = inc.get(stmt.target.id, 0) - step
        return inc

    def _expand_transformed_image_pattern(pattern: str, L: int, for_node: Optional[ast.For]) -> Optional[List[str]]:
        """
        If pattern is transformed_image_{i}.png and:
          - i is assigned a constant before the loop
          - loop body contains i += 1 (or i -= 1) with constant step
          - loop length L is known
        then generate concrete names.
        """
        if L <= 1 or not for_node:
            return None

        m = re.fullmatch(r"transformed_image_\{([a-zA-Z_]\w*)\}\.png", pattern)
        if not m:
            return None

        var = m.group(1)
        if var not in var_const_int:
            return None

        inc = _loop_inc_vars(for_node)
        step = inc.get(var)
        if step is None:
            return None

        start = var_const_int[var]
        names = [f"transformed_image_{start + step * t}.png" for t in range(L)]
        # rough simulate update after loop
        var_const_int[var] = start + step * L
        return names

    class V(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            if node.targets and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id

                # container length tracking
                L = _static_iter_len(node.value, var_len)
                if L is not None:
                    var_len[var_name] = L

                # constant int assignment (i = 2)
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                    var_const_int[var_name] = node.value.value

                # filename pattern assignment
                filename = _extract_filename_pattern(node.value, var_to_filename)
                if filename and (_IMG_RE.search(filename) or _ANY_IMG_RE.search(filename)):
                    var_to_filename[var_name] = filename

            self.generic_visit(node)

        def visit_For(self, node: ast.For):
            L = _static_iter_len(node.iter, var_len)
            cur = loop_mult_stack[-1]
            if L is None:
                loop_mult_stack.append(cur)
                for_stack.append(node)
            else:
                loop_mult_stack.append(cur * L)
                for_stack.append(node)

            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)

            for_stack.pop()
            loop_mult_stack.pop()

        def visit_Call(self, node: ast.Call):
            nonlocal has_crop
            cur_mult = loop_mult_stack[-1]
            cur_for = for_stack[-1] if for_stack else None

            if isinstance(node.func, ast.Attribute):
                m = node.func.attr.lower()
                chain = _attr_chain(node.func).lower()
                is_cv2 = chain.startswith("cv2") or "opencv" in chain
                is_pil = not is_cv2

                # PIL crop
                if m == "crop" and is_pil:
                    _append_ops_n_times(
                        OpEvent(
                            op="crop",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "pil_method"},
                        ),
                        cur_mult,
                    )
                    has_crop = True

                # PIL rotate
                elif m == "rotate" and is_pil:
                    _append_ops_n_times(
                        OpEvent(
                            op="rotate",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "pil_method"},
                        ),
                        cur_mult,
                    )

                # PIL resize (only if no crop has been seen)
                elif m == "resize" and is_pil:
                    if not has_crop:
                        _append_ops_n_times(
                            OpEvent(
                                op="resize",
                                lineno=getattr(node, "lineno", 0),
                                col=getattr(node, "col_offset", 0),
                                detail={"kind": "pil_method"},
                            ),
                            cur_mult,
                        )

                # PIL enhance (ImageEnhance... .enhance())
                if m == "enhance" and is_pil:
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "pil_enhance"},
                        ),
                        cur_mult,
                    )

                # PIL transpose for flip/enhance
                if m == "transpose" and is_pil:
                    if node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.Attribute):
                            arg_name = arg.attr.upper()
                            if "FLIP" in arg_name:
                                _append_ops_n_times(
                                    OpEvent(
                                        op="flip",
                                        lineno=getattr(node, "lineno", 0),
                                        col=getattr(node, "col_offset", 0),
                                        detail={"kind": "pil_transpose", "method": arg_name},
                                    ),
                                    cur_mult,
                                )
                            else:
                                _append_ops_n_times(
                                    OpEvent(
                                        op="enhance",
                                        lineno=getattr(node, "lineno", 0),
                                        col=getattr(node, "col_offset", 0),
                                        detail={"kind": "pil_transpose", "method": arg_name},
                                    ),
                                    cur_mult,
                                )
                        else:
                            _append_ops_n_times(
                                OpEvent(
                                    op="flip",
                                    lineno=getattr(node, "lineno", 0),
                                    col=getattr(node, "col_offset", 0),
                                    detail={"kind": "pil_transpose"},
                                ),
                                cur_mult,
                            )

                # ImageDraw operations -> enhance
                if m in ("rectangle", "line", "ellipse", "polygon", "text", "point"):
                    if ("draw" in chain or "imagedraw" in chain or is_pil):
                        _append_ops_n_times(
                            OpEvent(
                                op="enhance",
                                lineno=getattr(node, "lineno", 0),
                                col=getattr(node, "col_offset", 0),
                                detail={"kind": "imagedraw", "method": m},
                            ),
                            cur_mult,
                        )

                # ImageFilter filter -> enhance
                if m == "filter" and ("imagefilter" in chain or is_pil):
                    filter_type = "unknown"
                    if node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.Attribute):
                            filter_type = arg.attr.lower()
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "imagefilter", "filter": filter_type},
                        ),
                        cur_mult,
                    )

                # enhance-like PIL methods
                if m in ("convert", "point", "paste", "putalpha", "putpixel", "autocontrast") and is_pil:
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "pil_enhance_method", "method": m},
                        ),
                        cur_mult,
                    )

                # PIL save
                if m == "save" and is_pil:
                    for a in node.args:
                        filename = _extract_filename_pattern(a, var_to_filename)
                        if not filename and isinstance(a, ast.Name):
                            filename = var_to_filename.get(a.id, "")

                        if not filename:
                            continue

                        L = _static_iter_len(cur_for.iter, var_len) if cur_for else None

                        # Try to concretize transformed_image_{i}.png into [2,3,4,...]
                        if L and L > 1:
                            expanded = _expand_transformed_image_pattern(filename, L, cur_for)
                            if expanded:
                                for fname in expanded:
                                    mm = _IMG_RE.search(fname)
                                    if mm:
                                        k = int(mm.group(1))
                                        saves_raw.append(
                                            (
                                                k,
                                                getattr(node, "lineno", 0),
                                                getattr(node, "col_offset", 0),
                                                f"transformed_image_{k}.png",
                                                True,
                                            )
                                        )
                                    elif _ANY_IMG_RE.search(fname):
                                        saves_raw.append(
                                            (
                                                -1,
                                                getattr(node, "lineno", 0),
                                                getattr(node, "col_offset", 0),
                                                fname,
                                                False,
                                            )
                                        )
                                continue

                        # Normal: expand by loop multiplier, keep pattern if not concrete
                        mm = _IMG_RE.search(filename)
                        if mm:
                            k = int(mm.group(1))
                            _append_save_raw_n_times(
                                (
                                    k,
                                    getattr(node, "lineno", 0),
                                    getattr(node, "col_offset", 0),
                                    f"transformed_image_{k}.png",
                                    True,
                                ),
                                cur_mult,
                            )
                        elif _ANY_IMG_RE.search(filename):
                            _append_save_raw_n_times(
                                (
                                    -1,
                                    getattr(node, "lineno", 0),
                                    getattr(node, "col_offset", 0),
                                    filename,
                                    False,
                                ),
                                cur_mult,
                            )

            # ----------------------------
            # OpenCV branch (FULL, not omitted)
            # ----------------------------
            if isinstance(node.func, ast.Attribute):
                chain = _attr_chain(node.func).lower()

                # OpenCV resize
                if chain.endswith("cv2.resize"):
                    _append_ops_n_times(
                        OpEvent(
                            op="resize",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2"},
                        ),
                        cur_mult,
                    )

                # OpenCV rotation / affine / perspective
                if chain.endswith("cv2.warpaffine") or chain.endswith("cv2.getrotationmatrix2d"):
                    _append_ops_n_times(
                        OpEvent(
                            op="rotate",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2"},
                        ),
                        cur_mult,
                    )
                if chain.endswith("cv2.warpperspective"):
                    _append_ops_n_times(
                        OpEvent(
                            op="rotate",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2_perspective"},
                        ),
                        cur_mult,
                    )

                # OpenCV flip
                if chain.endswith("cv2.flip"):
                    _append_ops_n_times(
                        OpEvent(
                            op="flip",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2_flip"},
                        ),
                        cur_mult,
                    )

                # OpenCV drawing -> enhance
                if chain.endswith(("cv2.rectangle", "cv2.line", "cv2.circle", "cv2.ellipse", "cv2.polylines", "cv2.puttext")):
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2_draw"},
                        ),
                        cur_mult,
                    )

                # OpenCV edge -> enhance
                if chain.endswith(("cv2.canny", "cv2.sobel", "cv2.laplacian", "cv2.scharr")):
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2_edge"},
                        ),
                        cur_mult,
                    )

                # OpenCV filter -> enhance
                if chain.endswith(("cv2.blur", "cv2.gaussianblur", "cv2.medianblur", "cv2.bilateralfilter", "cv2.filter2d")):
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2_filter"},
                        ),
                        cur_mult,
                    )

                # OpenCV morph -> enhance
                if chain.endswith(("cv2.erode", "cv2.dilate", "cv2.morphologyex")):
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2_morph"},
                        ),
                        cur_mult,
                    )

                # OpenCV color/contrast -> enhance
                if chain.endswith(("cv2.cvtcolor", "cv2.equalizehist", "cv2.clahe", "cv2.convertscaleabs")):
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2_color"},
                        ),
                        cur_mult,
                    )

                # OpenCV threshold -> enhance
                if chain.endswith(("cv2.threshold", "cv2.adaptivethreshold")):
                    _append_ops_n_times(
                        OpEvent(
                            op="enhance",
                            lineno=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            detail={"kind": "cv2_threshold"},
                        ),
                        cur_mult,
                    )

                # OpenCV save
                if chain.endswith("cv2.imwrite"):
                    if node.args:
                        filename = _extract_filename_pattern(node.args[0], var_to_filename)
                        if not filename and isinstance(node.args[0], ast.Name):
                            filename = var_to_filename.get(node.args[0].id, "")

                        if filename:
                            mm = _IMG_RE.search(filename)
                            if mm:
                                k = int(mm.group(1))
                                _append_save_raw_n_times(
                                    (
                                        k,
                                        getattr(node, "lineno", 0),
                                        getattr(node, "col_offset", 0),
                                        f"transformed_image_{k}.png",
                                        True,
                                    ),
                                    cur_mult,
                                )
                            elif _ANY_IMG_RE.search(filename):
                                _append_save_raw_n_times(
                                    (
                                        -1,
                                        getattr(node, "lineno", 0),
                                        getattr(node, "col_offset", 0),
                                        filename,
                                        False,
                                    ),
                                    cur_mult,
                                )

            self.generic_visit(node)

    V().visit(tree)

    ops_sorted = sorted(ops, key=lambda x: (x.lineno, x.col))
    saves_sorted = sorted(saves_raw, key=lambda x: (x[1], x[2]))

    save_events: List[SaveEvent] = []
    prev_save_line = 0
    non_standard_idx = 0

    for idx, lineno, col, fname, is_standard in saves_sorted:
        if idx == -1:
            actual_idx = non_standard_idx
            non_standard_idx += 1
        else:
            actual_idx = idx

        ops_in_range = [o for o in ops_sorted if prev_save_line < o.lineno <= lineno]

        # Priority: crop > flip > rotate > resize > enhance
        op_guess = "unknown"
        for o in ops_in_range:
            if o.op == "crop":
                op_guess = "crop"
                break
        if op_guess == "unknown":
            for o in ops_in_range:
                if o.op == "flip":
                    op_guess = "flip"
                    break
        if op_guess == "unknown":
            for o in ops_in_range:
                if o.op == "rotate":
                    op_guess = "rotate"
                    break
        if op_guess == "unknown":
            for o in ops_in_range:
                if o.op == "resize":
                    op_guess = "resize"
                    break
        if op_guess == "unknown" and ops_in_range:
            op_guess = ops_in_range[-1].op

        if op_guess == "unknown":
            ops_before_save = [o for o in ops_sorted if o.lineno < lineno]
            if ops_before_save:
                for o in ops_before_save:
                    if o.op == "crop":
                        op_guess = "crop"
                        break
                if op_guess == "unknown":
                    for o in ops_before_save:
                        if o.op == "flip":
                            op_guess = "flip"
                            break
                if op_guess == "unknown":
                    for o in ops_before_save:
                        if o.op == "rotate":
                            op_guess = "rotate"
                            break
                if op_guess == "unknown":
                    for o in ops_before_save:
                        if o.op == "resize":
                            op_guess = "resize"
                            break
                if op_guess == "unknown":
                    op_guess = ops_before_save[-1].op

        save_events.append(
            SaveEvent(
                image_index=actual_idx,
                lineno=lineno,
                col=col,
                filename=fname,
                op_guess=op_guess,
                is_standard_name=is_standard,
            )
        )
        prev_save_line = lineno

    return ops_sorted, save_events


# ---------------------------
# NEW: build tool-use events for UI (ALL ops, not just op_guess)
# ---------------------------
def infer_tool_events(code: str) -> List[Dict[str, Any]]:
    """
    Return a flat list of tool events for UI:
    - Every op occurrence becomes one tool event (crop/enhance/rotate/resize/flip...)
    - Each op is associated with the nearest following save in its segment (prev_save -> save),
      so crop+enhance+enhance -> save will produce 3 tool events with same save filename.
    """
    ops, saves = infer_ops_and_saves(code)

    # save points in order
    save_points = [(s.lineno, s.col, s.filename) for s in saves]
    save_points.sort()

    tool_events: List[Dict[str, Any]] = []

    # For each op, find next save at/after op
    for o in ops:
        save_name = ""
        for (sl, sc, fn) in save_points:
            if (sl > o.lineno) or (sl == o.lineno and sc >= o.col):
                save_name = fn
                break

        tool_events.append(
            {
                "tool_name": o.op,
                "arguments": {
                    "op": o.op,
                    "save": save_name,
                    "line": o.lineno,
                },
                "detail": o.detail,
            }
        )

    return tool_events