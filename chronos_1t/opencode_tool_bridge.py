"""OpenCode / OpenAI Agentic Tool-Calling Bridge for DeepSeek-V4.1-Flash-1017 & Chronos v7.5.

Ports and unifies the multi-format tool-call parser, JSON-Schema parameter type coercer,
and chat-template message normalizer from Chimera-LSA (`colab_worker.py` / `serve_opencode.py`)
with native DeepSeek-V4.1 `<｜tool▁calls▁begin｜>` token support.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any


def normalize_messages_for_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalizes OpenAI / OpenCode multi-turn messages (including array content blocks
    and stringified `tool_calls.function.arguments`) for Jinja chat templates.
    """
    norm: list[dict[str, Any]] = []
    for m in messages:
        m_copy = dict(m)
        content = m_copy.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif isinstance(p, str):
                    parts.append(p)
            m_copy["content"] = "\n".join(parts)
        elif content is None:
            m_copy["content"] = ""
        if "tool_calls" in m_copy and isinstance(m_copy["tool_calls"], list):
            fixed_tcs: list[dict[str, Any]] = []
            for tc in m_copy["tool_calls"]:
                tc_copy = dict(tc)
                fn = dict(tc_copy.get("function", {}))
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except Exception:
                        fn["arguments"] = {}
                tc_copy["function"] = fn
                fixed_tcs.append(tc_copy)
            m_copy["tool_calls"] = fixed_tcs
        norm.append(m_copy)
    return norm


def coerce_param_value(val_str: str, param_schema: dict[str, Any] | None = None) -> Any:
    """Coerces a raw string parameter value to the JSON-Schema target type
    (`integer`, `number`, `boolean`, `array`, `object`, or `string`).
    """
    raw = val_str.strip()
    target_type = (param_schema or {}).get("type")
    if isinstance(target_type, list):
        non_null = [t for t in target_type if t != "null"]
        target_type = non_null[0] if non_null else "string"

    if target_type == "string":
        return val_str.strip("\n")
    if target_type == "integer":
        try:
            return int(raw)
        except Exception:
            pass
    if target_type == "number":
        try:
            return float(raw)
        except Exception:
            pass
    if target_type == "boolean":
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False
    if target_type in ("object", "array"):
        try:
            return json.loads(raw)
        except Exception:
            pass
    return val_str.strip("\n")


def coerce_args_dict(args_dict: dict[str, Any], fn_props: dict[str, Any]) -> dict[str, Any]:
    """Applies JSON-Schema type coercion across an already-parsed arguments dict."""
    if not isinstance(args_dict, dict) or not fn_props:
        return args_dict
    out: dict[str, Any] = {}
    for k, v in args_dict.items():
        if isinstance(v, str) and k in fn_props:
            out[k] = coerce_param_value(v, fn_props.get(k))
        else:
            out[k] = v
    return out


_DSV41_TC_BLOCK_RE = re.compile(
    r"<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>",
    re.DOTALL,
)
_DSV41_SINGLE_TC_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?:(?P<type>[^\n<｜]+)<｜tool▁sep｜>)?(?P<name>[^\n<｜]+)\n(?:```(?:json)?\n)?(?P<args>.*?)(?:\n```)?<｜tool▁call▁end｜>",
    re.DOTALL,
)
_XML_FN_TC_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    re.DOTALL,
)
_JSON_TC_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls_and_clean_content(
    text: str,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Extracts structured OpenAI-compatible `tool_calls` from model output and strips
    the raw tool markup from `content`.

    Supports three formats seamlessly:
      1. DeepSeek-V4.1 native: `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>name\\n```json\\n{...}\\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
      2. LFM3 / XML parameter tags: `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
      3. Qwen / Hermes JSON tags: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
    """
    tool_schemas: dict[str, dict[str, Any]] = {}
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        name = fn.get("name")
        if name:
            tool_schemas[name] = (fn.get("parameters") or {}).get("properties") or {}

    tool_calls: list[dict[str, Any]] = []

    # 1. DeepSeek-V4.1 native tool call blocks
    for m_ds in _DSV41_SINGLE_TC_RE.finditer(text):
        fn_name = m_ds.group("name").strip()
        raw_args = m_ds.group("args").strip()
        fn_props = tool_schemas.get(fn_name, {})
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
        except Exception:
            parsed_args = {"raw": raw_args}
        if isinstance(parsed_args, dict):
            parsed_args = coerce_args_dict(parsed_args, fn_props)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(parsed_args, ensure_ascii=False),
                },
            }
        )

    # 2. LFM3 / XML `<function=...><parameter=...>` blocks
    for m_xml in _XML_FN_TC_RE.finditer(text):
        fn_name = m_xml.group(1).strip()
        body = m_xml.group(2)
        fn_props = tool_schemas.get(fn_name, {})
        args_dict: dict[str, Any] = {}
        for pm in _XML_PARAM_RE.finditer(body):
            p_name = pm.group(1).strip()
            p_val = pm.group(2)
            args_dict[p_name] = coerce_param_value(p_val, fn_props.get(p_name))
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(args_dict, ensure_ascii=False),
                },
            }
        )

    # 3. Qwen / Hermes JSON `<tool_call>{...}</tool_call>` blocks
    if not tool_calls:
        for m_js in _JSON_TC_RE.finditer(text):
            try:
                obj = json.loads(m_js.group(1))
                fn_name = obj.get("name", "unknown")
                args_obj = obj.get("arguments", {})
                if isinstance(args_obj, str):
                    try:
                        args_obj = json.loads(args_obj)
                    except Exception:
                        pass
                if isinstance(args_obj, dict):
                    args_obj = coerce_args_dict(args_obj, tool_schemas.get(fn_name, {}))
                args_str = args_obj if isinstance(args_obj, str) else json.dumps(args_obj, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_str,
                        },
                    }
                )
            except Exception:
                pass

    cleaned = _DSV41_TC_BLOCK_RE.sub("", text)
    cleaned = _DSV41_SINGLE_TC_RE.sub("", cleaned)
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL).strip()
    return cleaned, tool_calls
