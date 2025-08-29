
from __future__ import annotations
import json, re, random, ast, unicodedata, datetime as dt, math
from typing import Any, Dict, List, Tuple, Optional

# Optional tolerant parsers
TRY_LIBS = {}
for lib in ("pyjson5", "tolerantjson", "demjson3"):
    try:
        TRY_LIBS[lib] = __import__(lib)
    except Exception:
        TRY_LIBS[lib] = None

SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2018": "'", "\u2019": "'", "\u2032": "'",
}

def clean_common(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = "".join(SMART_QUOTES.get(ch, ch) for ch in text)
    text = text.lstrip("\ufeff")
    return text

def replace_python_literals(text: str) -> str:
    return re.sub(r"\b(True|False|None)\b", lambda m: {"True":"true","False":"false","None":"null"}[m.group(1)], text)

def fix_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)

def quote_unquoted_keys(text: str) -> str:
    pattern = re.compile(r'([{\s,])([A-Za-z_][\w\-]*)\s*:')
    return pattern.sub(lambda m: f'{m.group(1)}"{m.group(2)}":', text)

def to_double_quotes(text: str) -> str:
    text = re.sub(r"'([A-Za-z_][^']*)'\s*:", r'"\1":', text)
    text = re.sub(r':\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'', r': "\1"', text)
    text = re.sub(r'\[\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'', r'[ "\1"', text)
    text = re.sub(r',\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'', r', "\1"', text)
    return text

def replace_nan_infinity(text: str) -> str:
    return re.sub(r'\b(NaN|Infinity|-Infinity)\b', 'null', text)

def add_missing_commas(text: str) -> str:
    text = re.sub(r'\}\s*\{', '},{', text)
    text = re.sub(r'(":\s*[^,{}\[\]]+)\s*(")', r'\1,\2', text)
    return text

def add_missing_commas_more(text: str) -> str:
    text = re.sub(r'(":\s*"[^"\\]*(?:\\.[^"\\]*)*")\s*(")', r'\1,\2', text)
    text = re.sub(r'(":\s*(?:-?\d+(?:\.\d+)?|true|false|null))\s*(")', r'\1,\2', text, flags=re.IGNORECASE)
    return text

def escape_unescaped_newlines(text: str) -> str:
    return text.replace("\n", "\\n")

def repair_invalid_unicode(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def repair_unbalanced_braces(text: str) -> str:
    opens = text.count("{"); closes = text.count("}")
    if opens > closes and opens - closes < 5:
        text = text + "}" * (opens - closes)
    opens = text.count("["); closes = text.count("]")
    if opens > closes and opens - closes < 5:
        text = text + "]" * (opens - closes)
    return text

def try_parsers(text: str):
    try:
        return json.loads(text), "json"
    except Exception:
        pass
    if TRY_LIBS.get("pyjson5"):
        try:
            return TRY_LIBS["pyjson5"].loads(text), "pyjson5"
        except Exception:
            pass
    if TRY_LIBS.get("tolerantjson"):
        try:
            return TRY_LIBS["tolerantjson"].parse(text), "tolerantjson"
        except Exception:
            pass
    if TRY_LIBS.get("demjson3"):
        try:
            return TRY_LIBS["demjson3"].decode(text), "demjson3"
        except Exception:
            pass
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (dict, list)):
            return obj, "ast.literal_eval"
    except Exception:
        pass
    return None, None

def extract_json_blocks(text: str) -> List[str]:
    blocks, stack = [], []
    in_str = False
    esc = False
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "\\" and in_str:
            esc = not esc
            continue
        if ch in ('"', "'"):
            if not in_str:
                in_str = ch
            elif in_str == ch and not esc:
                in_str = False
            esc = False
        elif not in_str:
            if ch in "{[":
                if not stack:
                    start_idx = i
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        blocks.append(text[start_idx : i + 1])
                        start_idx = None
        else:
            esc = False
    return blocks

def remove_preface_epilogue(text: str) -> str:
    blocks = extract_json_blocks(text)
    if blocks:
        return max(blocks, key=len)
    return text

def post_parse_normalize(obj):
    if isinstance(obj, dict):
        return {k: post_parse_normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [post_parse_normalize(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj

def truncate_to_last_valid_json(text: str, max_backtrack: int = 5000):
    for cut in range(0, min(len(text), max_backtrack)):
        candidate = text[: len(text) - cut]
        candidate = repair_unbalanced_braces(candidate)
        obj, _ = try_parsers(candidate)
        if obj is not None:
            return candidate
    return None

def parse_and_fix(raw: str):
    debug = {"attempts": []}
    obj, parser = try_parsers(raw)
    if obj is not None:
        debug["attempts"].append(f"parsed_with={parser}")
        return post_parse_normalize(obj), debug

    s = raw
    for step_name, fn in [
        ("clean_common", clean_common),
        ("replace_python_literals", replace_python_literals),
        ("replace_nan_infinity", replace_nan_infinity),
        ("fix_trailing_commas", fix_trailing_commas),
        ("add_missing_commas_more", add_missing_commas_more),
        ("to_double_quotes", to_double_quotes),
        ("quote_unquoted_keys", quote_unquoted_keys),
        ("add_missing_commas", add_missing_commas),
        ("remove_preface_epilogue", remove_preface_epilogue),
        ("escape_unescaped_newlines", escape_unescaped_newlines),
        ("repair_invalid_unicode", repair_invalid_unicode),
        ("repair_unbalanced_braces", repair_unbalanced_braces),
    ]:
        s = fn(s)
        obj, parser = try_parsers(s)
        debug["attempts"].append(step_name)
        if obj is not None:
            debug["attempts"].append(f"parsed_with={parser}")
            return post_parse_normalize(obj), debug

    block = remove_preface_epilogue(raw)
    if block != raw:
        s = repair_unbalanced_braces(fix_trailing_commas(replace_python_literals(replace_nan_infinity(to_double_quotes(quote_unquoted_keys(add_missing_commas_more(block)))))))
        obj, parser = try_parsers(s)
        debug["attempts"].append("final_block_retry")
        if obj is not None:
            debug["attempts"].append(f"parsed_with={parser}")
            return post_parse_normalize(obj), debug

    candidate = truncate_to_last_valid_json(raw)
    if candidate is not None:
        obj, parser = try_parsers(candidate)
        debug["attempts"].append("truncate_to_last_valid_json")
        if obj is not None:
            debug["attempts"].append(f"parsed_with={parser}")
            return post_parse_normalize(obj), debug

    raise ValueError("Unable to repair JSON")

Schema = Dict[str, Any]

def coerce_basic_types(value: Any, type_str: str):
    try:
        if type_str == "string":
            if isinstance(value, (int, float, bool)):
                return str(value)
            return "" if value is None else str(value)
        if type_str == "integer":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float, str)):
                return int(float(value))
        if type_str == "number":
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.replace(",", ""))
        if type_str == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("true","1","yes","y")
            if isinstance(value, (int, float)):
                return value != 0
        if type_str == "array":
            return list(value) if not isinstance(value, list) else value
        if type_str == "object":
            return dict(value) if not isinstance(value, dict) else value
    except Exception:
        return None
    return value

def coerce_to_schema(data: Any, schema: Schema) -> Any:
    t = schema.get("type")
    if t == "object" and isinstance(data, list):
        data = data[0] if data else {}
    if t == "array" and isinstance(data, dict):
        data = [data]
    if t == "object" and isinstance(data, dict):
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional = schema.get("additionalProperties", False)
        out = {}
        for k, subschema in props.items():
            if k in data:
                out[k] = coerce_to_schema(data[k], subschema)
            elif "default" in subschema:
                out[k] = subschema["default"]
            elif k in required:
                out[k] = None
        for k in list(out.keys()):
            if out[k] is None and k not in required:
                out.pop(k, None)
        if additional is True:
            for k, v in data.items():
                if k not in out:
                    out[k] = v
        return out
    elif t == "array" and isinstance(data, list):
        item_schema = schema.get("items", {})
        return [coerce_to_schema(x, item_schema) for x in data]
    elif t in {"string","integer","number","boolean"}:
        return coerce_basic_types(data, t)
    else:
        coerced = coerce_basic_types(data, t) if t else data
        return coerced

if __name__ == "__main__":
    # Example usage
    example_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "score": {"type": "number"},
            "active": {"type": "boolean"},
            "created_at": {"type": "string"},
        },
        "required": ["name", "email"],
        "additionalProperties": False,
    }
    bad = "```json\\n{'name': 'Jane', 'age': '32', 'email': 'jane@ex.com', 'tags': ['a','b'], 'score': NaN, 'active': True}\\n```"
    obj, dbg = parse_and_fix(bad)
    coerced = coerce_to_schema(obj, example_schema)
    print(coerced)
    print(dbg)


# in llm_json_fixer.py

def process_output_with_context(raw: str, *, prefer_jsonl: bool = True) -> dict:
    """
    Robustly handle outputs that include extra text around JSON (preface/notes) or JSONL.

    Returns:
      {
        "data": dict|list|list[dict]|None,
        "data_format": "object"|"array"|"jsonl"|"unknown",
        "extra_text": "<non-json text>",
        "other_json_candidates": [ "<json str>", ... ],
        "debug": {...}
      }
    Strategy (in order):
      1) Detect JSONL (>=2 parsable JSON lines) and strip used lines as data.
      2) Else, extract JSON blocks; parse the largest; return the rest as extra.
      3) Else, whole-text repair.
    """
    import json
    info = {"debug": {"steps": []}}
    text = raw

    # ---------- local helpers (no external dependencies) ----------
    def _looks_like_json_object_or_array(line: str) -> bool:
        s = line.strip()
        return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

    def _split_jsonl_candidates(txt: str):
        return [ln for ln in txt.splitlines() if ln.strip()]

    def _extract_json_blocks(txt: str):
        """Stack-based extraction; ignores braces inside quoted strings."""
        blocks, stack = [], []
        in_str = None
        esc = False
        start = None
        for i, ch in enumerate(txt):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
            else:
                if ch in ("\"", "'"):
                    in_str = ch
                elif ch in "{[":
                    if not stack:
                        start = i
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        stack.pop()
                        if not stack and start is not None:
                            blocks.append(txt[start:i+1])
                            start = None
        return blocks
    # ---------------------------------------------------------------

    # 1) JSONL detection first (most robust for multi-record outputs)
    if prefer_jsonl:
        lines = _split_jsonl_candidates(text)
        objs, used_lines, ok = [], [], 0
        for ln in lines:
            if not _looks_like_json_object_or_array(ln):
                continue
            try:
                o, _ = parse_and_fix(ln)  # <- your deterministic fixer
                objs.append(o)
                used_lines.append(ln)
                ok += 1
            except Exception:
                pass
        if ok >= 2:  # likely JSONL
            used_text = "\n".join(used_lines)
            extra_text = text.replace(used_text, "", 1)
            info["debug"]["steps"].append({"attempt": "jsonl_detect", "ok_lines": ok})
            return {
                "data": objs,
                "data_format": "jsonl",
                "extra_text": extra_text.strip(),
                "other_json_candidates": [],
                "debug": info["debug"],
            }

    # 2) Extract explicit JSON blocks and choose the largest parsable
    blocks = _extract_json_blocks(text)
    info["debug"]["steps"].append({"attempt": "extract_json_blocks", "count": len(blocks)})
    best_obj = best_block = None
    for blk in sorted(blocks, key=len, reverse=True):
        try:
            o, dbg = parse_and_fix(blk)
            best_obj, best_block = o, blk
            info["debug"]["steps"].append({"attempt": "choose_best_block", "notes": dbg.get("attempts", [])})
            break
        except Exception:
            continue
    if best_obj is not None:
        extra = text.replace(best_block, "", 1).strip()
        fmt = "object" if isinstance(best_obj, dict) else "array" if isinstance(best_obj, list) else "unknown"
        others = [b for b in blocks if b is not best_block]
        return {
            "data": best_obj,
            "data_format": fmt,
            "extra_text": extra,
            "other_json_candidates": others,
            "debug": info["debug"],
        }

    # 3) Last-resort whole-text repair (may merge pieces)
    try:
        o, dbg = parse_and_fix(text)
        fmt = "object" if isinstance(o, dict) else "array" if isinstance(o, list) else "unknown"
        info["debug"]["steps"].append({"attempt": "parse_and_fix(full_last)", "ok": True, "notes": dbg.get("attempts", [])})
        return {"data": o, "data_format": fmt, "extra_text": "", "other_json_candidates": [], "debug": info["debug"]}
    except Exception as e:
        info["debug"]["steps"].append({"attempt": "parse_and_fix(full_last)", "ok": False, "error": str(e)})

    # Give up — hand back everything as extra text
    return {"data": None, "data_format": "unknown", "extra_text": text.strip(), "other_json_candidates": [], "debug": info["debug"]}


def parse_additional_info(extra_text: str) -> dict:
    """
    Parse simple 'key: value' lines from extra text (e.g., Notes, Model, Confidence).
    Matches '- key: value' or 'key: value'. Returns a dict of lowercased keys.
    """
    info = {}
    for ln in extra_text.splitlines():
        ln = ln.strip(" -\t")
        if not ln or ln.lower().endswith(":"):
            continue
        if ":" in ln:
            k, v = ln.split(":", 1)
            k = k.strip().lower(); v = v.strip()
            if k:
                info[k] = v
    return info
