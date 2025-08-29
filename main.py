# robust_json_pipeline.py
from __future__ import annotations
import os, json, random, re, datetime as dt
from typing import Any, Dict, List, Tuple

# 1) deterministic fixer from your earlier module
from llm_json_fixer import parse_and_fix  # provided earlier
from llm_json_fixer import process_output_with_context, parse_additional_info

# 2) LangChain + Pydantic schema for LLM fallback
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser  # optional alt. :contentReference[oaicite:4]{index=4}
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError  # v1 shim used by LangChain :contentReference[oaicite:5]{index=5}

# ---------- Define your schema as a Pydantic model ----------
class Record(BaseModel):
    name: str
    email: str
    age: int | None = None
    tags: List[str] = []
    score: float | None = None
    active: bool | None = None
    created_at: str | None = None

# Parser over the schema
PD_PARSER = PydanticOutputParser(pydantic_object=Record)  # :contentReference[oaicite:6]{index=6}

def get_llm_for_fix() -> ChatOpenAI | None:
    """Return an LLM for fixing if API key is present; else None."""
    if os.getenv("OPENAI_API_KEY"):
        # Any provider supported by LangChain works; using OpenAI here.
        # For models/providers that support structured outputs natively,
        # see LangChain's with_structured_output pattern. :contentReference[oaicite:7]{index=7}
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return None

def heuristics_then_validate(raw: str) -> tuple[Record, dict]:
    """Deterministic repair + schema validation (no LLM)."""
    obj, dbg = parse_and_fix(raw)
    # Coerce using pydantic (raises on missing/wrong types). :contentReference[oaicite:8]{index=8}
    rec = Record.parse_obj(obj)  # returns a Record or raises ValidationError
    return rec, {"path": "heuristic", "attempts": dbg.get("attempts", [])}

def llm_fix_then_validate(raw: str, original_error: str) -> tuple[Record, dict]:
    """Use LangChain's OutputFixingParser (LLM) to repair, then validate."""
    llm = get_llm_for_fix()
    if not llm:
        raise RuntimeError("LLM fallback requested but OPENAI_API_KEY not set")

    # Strategy A: OutputFixingParser (give the broken text + schema) :contentReference[oaicite:9]{index=9}
    fix_parser = OutputFixingParser.from_llm(parser=PD_PARSER, llm=llm)  # :contentReference[oaicite:10]{index=10}
    try:
        fixed = fix_parser.parse(raw)  # returns Record (pydantic) on success
        return fixed, {"path": "llm_fix(OutputFixingParser)"}
    except Exception:
        # Strategy B: RetryWithErrorOutputParser – pass the error to the LLM :contentReference[oaicite:11]{index=11}
        retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=PD_PARSER, llm=llm, max_retries=2
        )
        fixed = retry_parser.parse_with_prompt(
            completion=raw,
            prompt="The output above failed to parse; repair it to match the schema strictly.",
            error=original_error,
        )
        return fixed, {"path": "llm_fix(RetryWithErrorOutputParser)"}

def robust_parse(raw: str) -> tuple[Record, dict]:
    """
    Detect + repair unknown JSON issues.
    Order:
      1) deterministic repairs (regex/heuristics, tolerant readers inside parse_and_fix)
      2) Pydantic validation
      3) LLM-based repair if needed
    """
    try:
        return heuristics_then_validate(raw)
    except (ValueError, ValidationError) as e:
        # Fall back to LLM only when absolutely necessary
        return llm_fix_then_validate(raw, original_error=str(e))

# ------------------- Synthetic scenarios for testing -------------------
COMMON_FAILURES = [
    "wrapped_in_markdown_code_fence",
    "preface_and_epilogue_text",
    "single_quotes_instead_of_double",
    "python_bools_none",
    "smart_quotes",
    "trailing_commas",
    "missing_commas_between_pairs",
    "unquoted_keys",
    "nan_and_infinity",
    "multiple_json_objects_concatenated",
    "json_inside_text_with_other_braces",
    "mixed_array_or_object_when_schema_expects_other",
    "numbers_as_strings",
    "date_format_variants",
    "duplicate_keys",
    "newline_in_string_unescaped",
    "invalid_unicode_surrogates",
    "truncated_output_missing_closing_brace",
]

def random_name():
    first = ["Alice","Bob","Chen","Dinesh","Elena","Fatima","Gaurav","Hiro","Ivy","Jun"]
    last  = ["Tan","Wong","Lim","Singh","Garcia","Kim","Nguyen","Kumar","Zhang","Ivanov"]
    return f"{random.choice(first)} {random.choice(last)}"

def base_valid_obj() -> Dict[str, Any]:
    return {
        "name": random_name(),
        "age": random.randint(18, 75),
        "email": "user@example.com",
        "tags": ["alpha", "beta", "gamma"],
        "score": round(random.random() * 100, 2),
        "active": random.choice([True, False]),
        "created_at": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

def mutate(obj: Dict[str, Any], scenario: str) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    if scenario == "wrapped_in_markdown_code_fence": return f"```json\n{s}\n```"
    if scenario == "preface_and_epilogue_text":     return f"Here you go:\n{s}\nHope that helps!"
    if scenario == "single_quotes_instead_of_double": return s.replace('"', "'")
    if scenario == "python_bools_none":             return s.replace("true","True").replace("false","False").replace("null","None")
    if scenario == "smart_quotes":                  return s.translate(str.maketrans({'"':'”', "'":"’"}))
    if scenario == "trailing_commas":               return re.sub(r'\]', r',]', re.sub(r'\}', r',}', s, count=1), count=1)
    if scenario == "missing_commas_between_pairs":  return re.sub(r',(\s*")', r'\1', s, count=1)
    if scenario == "unquoted_keys":                 return re.sub(r'"(\w+)"\s*:', r'\1:', s, count=1)
    if scenario == "nan_and_infinity":              return json.dumps({**obj, "score": float("nan"), "age": float("inf")}, ensure_ascii=False)
    if scenario == "multiple_json_objects_concatenated": return s + "\n" + json.dumps({**obj, "extra": True}, ensure_ascii=False)
    if scenario == "json_inside_text_with_other_braces": return f"BEGIN {{meta}}\nRESULT = {s}\nEND }}"
    if scenario == "mixed_array_or_object_when_schema_expects_other": return "[" + s + "]"
    if scenario == "numbers_as_strings":            return json.dumps({**obj, "age": str(obj["age"]), "score": str(obj["score"])}, ensure_ascii=False)
    if scenario == "date_format_variants":          return json.dumps({**obj, "created_at": dt.datetime.now().strftime("%d/%m/%Y %H:%M")}, ensure_ascii=False)
    if scenario == "duplicate_keys":                return s.replace('"tags": ["alpha", "beta", "gamma"]','"tags": ["alpha"], "tags": ["alpha","beta"]')
    if scenario == "newline_in_string_unescaped":   return json.dumps({**obj, "name": obj["name"] + "\nCEO"}, ensure_ascii=False)
    if scenario == "invalid_unicode_surrogates":    return s[:-1] + "\ud800" + s[-1:]
    if scenario == "truncated_output_missing_closing_brace": return s[:-5]
    raise ValueError(scenario)

def run_tests(seed: int = 42, n_per: int = 1):
    random.seed(seed)
    scenarios = []
    base = base_valid_obj()
    for sc in COMMON_FAILURES:
        for _ in range(n_per):
            scenarios.append((sc, mutate(base, sc)))

    passed_heur = 0
    passed_total = 0
    used_llm = 0
    results = []

    for sc, raw in scenarios:
        # Try heuristics only to see what would succeed without LLM
        try:
            rec, dbg = heuristics_then_validate(raw)
            passed_heur += 1
            passed_total += 1
            results.append((sc, "OK (heuristic)", dbg["attempts"][:4]))
            continue
        except Exception as e:
            # Now try LLM fallback (if available)
            try:
                rec, dbg = llm_fix_then_validate(raw, original_error=str(e))
                passed_total += 1
                used_llm += 1
                results.append((sc, "OK (LLM fallback)", [dbg["path"]]))
            except Exception as e2:
                results.append((sc, f"FAIL ({type(e2).__name__})", [str(e2)[:120]]))

    print("=== Summary ===")
    print({
        "total": len(scenarios),
        "passed_total": passed_total,
        "passed_heuristics_only": passed_heur,
        "repaired_by_llm": used_llm,
        "failed": len(scenarios) - passed_total
    })
    print("=== Details ===")
    for sc, status, notes in results:
        print(f"- {sc}: {status} | notes={notes}")


    # Example of processing output with context (JSONL + extra text)
    raw = """Here are the results:

    {"name":"Ava","email":"ava@example.com","age":"29","tags":["a","b"],"score": 88.2, "active": true}
    {"name":"Jun","email":"jun@example.com","age":"31","tags":["x"],"score": "77.5","active": false}

    Notes:
    - model: small-llm-1.1
    - confidence: medium
    """

    out = process_output_with_context(raw, prefer_jsonl=True)
    # out["data"] -> list[dict] (JSONL), out["extra_text"] -> the notes
    meta = parse_additional_info(out["extra_text"])


if __name__ == "__main__":
    run_tests()



