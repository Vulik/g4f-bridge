"""
Function Calling Emulator v4 — Robust parsing + strict normalization.
"""

import re
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from logger_setup import get_logger


class JSONExtractor:
    @staticmethod
    def extract(text: str) -> Optional[Dict[str, Any]]:
        if not text or not text.strip():
            return None

        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
            match = re.search(pattern, text)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue

        candidates = []
        i = 0
        while i < len(text):
            start = text.find('{', i)
            if start == -1:
                break

            depth = 0
            in_string = False
            escape = False
            j = start

            while j < len(text):
                c = text[j]
                if escape:
                    escape = False
                    j += 1
                    continue
                if c == '\\' and in_string:
                    escape = True
                    j += 1
                    continue
                if c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                candidate = json.loads(text[start:j + 1])
                                if isinstance(candidate, dict):
                                    candidates.append(candidate)
                            except (json.JSONDecodeError, ValueError):
                                pass
                            break
                j += 1
            i = start + 1

        for c in candidates:
            if "tool_calls" in c or "function_call" in c:
                return c
        for c in candidates:
            if "content" in c:
                return c
        if candidates:
            return candidates[0]

        return None


class FunctionCallingEmulator:
    FC_PROMPT = """You are a function-calling AI. Output ONLY valid JSON.

AVAILABLE FUNCTIONS:
{tools}

OUTPUT FORMAT:
{{"tool_calls":[{{"function":{{"name":"FUNCTION_NAME","arguments":{{"param":"value"}}}}}}]}}

EXAMPLES:
User: "Weather in Tokyo?" → {{"tool_calls":[{{"function":{{"name":"get_weather","arguments":{{"location":"Tokyo"}}}}}}]}}
User: "Search Python" → {{"tool_calls":[{{"function":{{"name":"web_search","arguments":{{"query":"Python"}}}}}}]}}
User: "Free RAM?" → {{"tool_calls":[{{"function":{{"name":"get_free_ram","arguments":{{}}}}}}]}}

RULES:
- Output ONLY JSON, nothing else
- Use "function" with "name" and "arguments"
- "arguments" must be {{}} object"""

    FC_RETRY_PROMPT = """OUTPUT ONLY:
{{"tool_calls":[{{"function":{{"name":"{fname}","arguments":{args}}}}}]}}"""

    def __init__(self) -> None:
        self.logger = get_logger("function_calling")

    def has_tools(self, body: Dict[str, Any]) -> bool:
        tools = body.get("tools")
        if not tools:
            return False
        if body.get("tool_choice") == "none":
            return False
        return True

    def has_tool_results(self, messages: List[Dict[str, str]]) -> bool:
        for msg in messages:
            if msg.get("role") == "tool":
                return True
        return False

    def build_messages(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "auto",
    ) -> Tuple[List[Dict[str, str]], str]:

        if self.has_tool_results(messages):
            self.logger.info("📥 Tool result detected")
            cleaned = self._clean_tool_messages(messages)
            return cleaned, "result"

        if tools:
            self.logger.info(f"🔧 FC Mode: {len(tools)} tools")

            tool_str = self._format_tools(tools)
            fc_prompt = self.FC_PROMPT.format(tools=tool_str)

            if tool_choice == "required":
                fc_prompt += "\n\nYou MUST call a function."
            elif isinstance(tool_choice, dict):
                fname = tool_choice.get("function", {}).get("name", "")
                if fname:
                    fc_prompt += f"\n\nCall: {fname}"

            enhanced = [{"role": "system", "content": fc_prompt}]

            for msg in messages:
                if msg.get("role") == "system":
                    original = msg.get("content", "")
                    if original and len(original) < 300:
                        enhanced[0]["content"] += f"\n\nContext: {original}"
                else:
                    enhanced.append(msg.copy())

            return enhanced, "fc"

        return messages, "chat"

    def build_retry_messages(
        self,
        original_messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        user_message: str,
    ) -> List[Dict[str, str]]:
        best = self._guess_best_function(user_message, tools)

        if best:
            func = best.get("function", {})
            fname = func.get("name", "unknown")
            props = func.get("parameters", {}).get("properties", {})
            args = {k: "..." for k in list(props.keys())[:2]} or {}
            retry = self.FC_RETRY_PROMPT.format(
                fname=fname, args=json.dumps(args)
            )
        else:
            retry = '{"tool_calls":[{"function":{"name":"...","arguments":{}}}]}'

        msgs = original_messages.copy()
        msgs.append({"role": "user", "content": retry})
        return msgs

    def _clean_tool_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        cleaned = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                cleaned.append({
                    "role": "user",
                    "content": f"[Tool Result for {tool_call_id}]:\n{content}\n\nRespond based on this.",
                })

            elif role == "assistant" and msg.get("tool_calls"):
                calls = msg.get("tool_calls", [])
                descs = []
                for tc in calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")
                    descs.append(f"{name}({args})")
                cleaned.append({
                    "role": "assistant",
                    "content": f"Called: {', '.join(descs)}",
                })

            else:
                cleaned.append(msg.copy())

        return cleaned

    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        lines = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "?")
            desc = func.get("description", "")[:80]
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            plist = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string")
                req = "*" if pname in required else ""
                plist.append(f"{pname}{req}:{ptype}")

            pstr = ", ".join(plist) if plist else ""
            lines.append(f"• {name}({pstr}) — {desc}")

        return "\n".join(lines)

    def parse_response(
        self,
        raw: str,
        tools: List[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, Any]:
        if not raw or not raw.strip():
            return {"type": "text", "tool_calls": None, "content": ""}

        if mode in ("result", "chat"):
            parsed = JSONExtractor.extract(raw)
            if parsed and "content" in parsed:
                return {
                    "type": "text",
                    "tool_calls": None,
                    "content": str(parsed["content"]),
                }
            clean = self._strip_json(raw)
            return {"type": "text", "tool_calls": None, "content": clean}

        parsed = JSONExtractor.extract(raw)

        if parsed:
            if "content" in parsed and "tool_calls" not in parsed:
                return {
                    "type": "text",
                    "tool_calls": None,
                    "content": str(parsed["content"]),
                }

            tool_calls = self._normalize_and_validate(parsed, tools)
            if tool_calls:
                names = [tc["function"]["name"] for tc in tool_calls]
                self.logger.info(f"✅ tool_calls: {', '.join(names)}")
                return {
                    "type": "tool_calls",
                    "tool_calls": tool_calls,
                    "content": None,
                }

        tool_calls = self._extract_from_text(raw, tools)
        if tool_calls:
            names = [tc["function"]["name"] for tc in tool_calls]
            self.logger.info(f"✅ Extracted: {', '.join(names)}")
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "content": None,
            }

        self.logger.warning(f"❌ No tool_calls: {raw[:80]}...")
        return {"type": "text", "tool_calls": None, "content": raw}

    def _normalize_and_validate(
        self,
        parsed: Dict[str, Any],
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tool_names = {
            t.get("function", {}).get("name", "").lower(): t.get("function", {}).get("name", "")
            for t in tools
        }

        raw_calls = []

        if "tool_calls" in parsed:
            tc = parsed["tool_calls"]
            raw_calls = tc if isinstance(tc, list) else [tc]
        elif "function_call" in parsed:
            raw_calls = [parsed["function_call"]]
        elif "function" in parsed and isinstance(parsed["function"], dict):
            raw_calls = [parsed]
        elif "name" in parsed:
            raw_calls = [parsed]
        elif "tool" in parsed:
            raw_calls = [parsed]

        if not raw_calls:
            return []

        result = []
        for raw in raw_calls:
            if not isinstance(raw, dict) or not raw:
                continue

            normalized = self._normalize_single(raw, tool_names)
            if normalized:
                result.append(normalized)

        return result

    def _normalize_single(
        self,
        raw: Dict[str, Any],
        tool_names: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        name = None
        arguments = {}

        if "function" in raw and isinstance(raw["function"], dict):
            name = raw["function"].get("name")
            arguments = raw["function"].get("arguments", {})
        elif "name" in raw:
            name = raw["name"]
            arguments = raw.get("arguments", raw.get("parameters", raw.get("params", {})))
        elif "tool" in raw:
            name = raw["tool"]
            arguments = {
                k: v for k, v in raw.items()
                if k not in ("tool", "id", "type", "index")
            }
        elif "function_name" in raw:
            name = raw["function_name"]
            arguments = raw.get("arguments", raw.get("parameters", {}))

        if not name:
            return None

        name_lower = str(name).lower().strip()
        matched = None

        if name_lower in tool_names:
            matched = tool_names[name_lower]
        else:
            for tn_lower, tn_orig in tool_names.items():
                if name_lower in tn_lower or tn_lower in name_lower:
                    matched = tn_orig
                    break
                norm_in = name_lower.replace("-", "_").replace(" ", "_")
                norm_tn = tn_lower.replace("-", "_").replace(" ", "_")
                if norm_in == norm_tn:
                    matched = tn_orig
                    break

        if not matched:
            return None

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {"input": arguments}

        if not isinstance(arguments, dict):
            arguments = {}

        cleaned = {}
        for k, v in arguments.items():
            if v not in ("?", "...", "<value>", "VALUE", None):
                cleaned[k] = v

        args_str = json.dumps(cleaned, ensure_ascii=False)

        return {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": matched,
                "arguments": args_str,
            },
        }

    def _extract_from_text(
        self,
        text: str,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not text or not tools:
            return []

        text_lower = text.lower()

        for tool in tools:
            func = tool.get("function", {})
            fname = func.get("name", "")
            fdesc = func.get("description", "").lower()
            params = func.get("parameters", {})
            required = params.get("required", [])
            props = params.get("properties", {})

            if not fname:
                continue

            fname_lower = fname.lower()
            words = fname_lower.replace("_", " ").split()

            matched = False

            if fname_lower in text_lower:
                matched = True
            elif all(w in text_lower for w in words if len(w) > 2):
                matched = True
            else:
                kws = [w for w in fdesc.split() if len(w) > 3 and w not in ("the", "for", "and", "with")]
                if kws and sum(1 for kw in kws if kw in text_lower) >= max(1, len(kws) // 2):
                    matched = True

            if not matched:
                continue

            arguments = {}
            for pname in props.keys():
                val = self._extract_param(text, pname, props[pname])
                if val:
                    arguments[pname] = val

            if required and not arguments:
                clean = re.sub(
                    r'^(please |can you |could you |i want |let me |search |find |get |what |how )',
                    '', text.strip(), flags=re.IGNORECASE
                ).strip()
                if clean and required:
                    arguments[required[0]] = clean

            return [{
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": fname,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }]

        return []

    def _extract_param(
        self,
        text: str,
        param_name: str,
        param_info: Dict[str, Any],
    ) -> Optional[str]:
        patterns = {
            "query": [r'(?:search|find)\s+(?:for\s+)?["\']?(.+?)["\']?(?:\.|$)'],
            "location": [r'(?:in|at|for)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|$|\?)'],
            "url": [r'(https?://\S+)'],
            "unit": [r'\b(MB|GB|KB|bytes?)\b'],
        }

        name_l = param_name.lower()
        desc_l = param_info.get("description", "").lower()

        for key, pats in patterns.items():
            if key in name_l or key in desc_l:
                for pat in pats:
                    m = re.search(pat, text, re.IGNORECASE)
                    if m:
                        return m.group(1).strip()

        quoted = re.findall(r'["\'](.+?)["\']', text)
        if quoted:
            return quoted[0]

        return None

    def _guess_best_function(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not tools:
            return None

        msg_l = user_message.lower()
        best = None
        best_score = 0

        for tool in tools:
            func = tool.get("function", {})
            fname = func.get("name", "").lower()
            fdesc = func.get("description", "").lower()

            score = 0
            for w in fname.replace("_", " ").split():
                if len(w) > 2 and w in msg_l:
                    score += 3
            for w in fdesc.split():
                if len(w) > 3 and w in msg_l:
                    score += 1

            if score > best_score:
                best_score = score
                best = tool

        return best or tools[0]

    def _strip_json(self, text: str) -> str:
        text = re.sub(r'```[\s\S]*?```', '', text)

        result = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 1
                j = i + 1
                while j < len(text) and depth > 0:
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                    j += 1
                i = j
            else:
                result.append(text[i])
                i += 1

        return ''.join(result).strip()

    def build_response(
        self,
        parsed: Dict[str, Any],
        model: str,
        prompt_tokens: int = 0,
    ) -> Dict[str, Any]:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if parsed["type"] == "tool_calls" and parsed["tool_calls"]:
            valid = [tc for tc in parsed["tool_calls"] if self._validate_tc(tc)]

            if valid:
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": valid,
                }
                finish = "tool_calls"
                comp = len(json.dumps(valid)) // 4
            else:
                message = {"role": "assistant", "content": ""}
                finish = "stop"
                comp = 1
        else:
            content = parsed.get("content") or ""
            message = {"role": "assistant", "content": content}
            finish = "stop"
            comp = max(1, len(content) // 4)

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp,
                "total_tokens": prompt_tokens + comp,
            },
        }

    def _validate_tc(self, tc: Dict[str, Any]) -> bool:
        if not isinstance(tc, dict):
            return False
        if "function" not in tc:
            return False
        func = tc["function"]
        if not isinstance(func, dict):
            return False
        if not func.get("name"):
            return False
        if "arguments" not in func:
            return False
        if not isinstance(func["arguments"], str):
            tc["function"]["arguments"] = json.dumps(func["arguments"])
        return True


_emulator: Optional[FunctionCallingEmulator] = None


def get_emulator() -> FunctionCallingEmulator:
    global _emulator
    if _emulator is None:
        _emulator = FunctionCallingEmulator()
    return _emulator