"""
Function Calling Emulator v2 — Simplified & Robust.
Handles full tool calling cycle including tool results.
PicoClaw compatible.
"""

import re
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from logger_setup import get_logger


class JSONExtractor:
    """Extract JSON from messy AI responses."""

    @staticmethod
    def extract(text: str) -> Optional[Dict[str, Any]]:
        if not text or not text.strip():
            return None

        text = text.strip()

        # Strategy 1: Whole text is JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: JSON in code blocks
        for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
            match = re.search(pattern, text)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue

        # Strategy 3: Find {...} with bracket matching
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        i = start

        while i < len(text):
            c = text[i]

            if escape:
                escape = False
                i += 1
                continue

            if c == '\\' and in_string:
                escape = True
                i += 1
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
                            return json.loads(text[start:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            next_start = text.find('{', start + 1)
                            if next_start != -1:
                                start = next_start
                                depth = 0
                                i = start
                                continue
                            return None
            i += 1

        return None


class FunctionCallingEmulator:
    """
    Handles full FC lifecycle:
    1. Detect if request needs FC
    2. Detect if request contains tool results
    3. Inject appropriate prompt
    4. Parse response
    5. Format output (PicoClaw compatible)
    """

    FC_PROMPT = """You are a function-calling AI. Respond ONLY with valid JSON.

Functions:
{tools}

To call a function:
{{"tool_calls":[{{"name":"FUNCTION_NAME","arguments":{{PARAMS}}}}]}}

To reply normally:
{{"content":"your reply"}}

RULES:
- Output ONLY JSON, nothing else
- NO markdown, NO explanation, NO backticks
- Pick ONE most relevant function
- Use EXACT function names"""

    def __init__(self) -> None:
        self.logger = get_logger("function_calling")

    # ── Detection ─────────────────────────────────────────

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

    def has_pending_tool_calls(self, messages: List[Dict[str, str]]) -> bool:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    return True
                break
        return False

    # ── Message Building ──────────────────────────────────

    def build_messages(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "auto",
    ) -> Tuple[List[Dict[str, str]], str]:
        # Case 1: Tool results present
        if self.has_tool_results(messages):
            self.logger.info("📥 Tool result detected — forwarding to AI")
            cleaned = self._clean_tool_messages(messages)
            return cleaned, "result"

        # Case 2: Tools requested, need FC
        if tools:
            self.logger.info(f"🔧 FC Mode: {len(tools)} tools")

            tool_str = self._format_tools_compact(tools)
            fc_prompt = self.FC_PROMPT.format(tools=tool_str)

            if tool_choice == "required":
                fc_prompt += "\nYou MUST call a function. Text-only response is FORBIDDEN."
            elif isinstance(tool_choice, dict):
                fname = tool_choice.get("function", {}).get("name", "")
                if fname:
                    fc_prompt += f"\nYou MUST call: {fname}"

            enhanced = [{"role": "system", "content": fc_prompt}]

            for msg in messages:
                if msg.get("role") == "system":
                    original = msg.get("content", "")
                    if original and len(original) < 500:
                        enhanced[0]["content"] += f"\n\nContext: {original}"
                else:
                    enhanced.append(msg.copy())

            return enhanced, "fc"

        # Case 3: Normal chat
        return messages, "chat"

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
                    "content": (
                        f"[Tool Result for {tool_call_id}]:\n{content}\n\n"
                        f"Based on this tool result, provide your response."
                    ),
                })

            elif role == "assistant" and msg.get("tool_calls"):
                calls = msg.get("tool_calls", [])
                call_desc = []
                for tc in calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")
                    call_desc.append(f"Called {name}({args})")

                cleaned.append({
                    "role": "assistant",
                    "content": "I called: " + ", ".join(call_desc),
                })

            else:
                cleaned.append(msg.copy())

        return cleaned

    def _format_tools_compact(
        self, tools: List[Dict[str, Any]]
    ) -> str:
        lines = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")[:60]
            params = func.get("parameters", {})
            required = params.get("required", [])

            if required:
                params_str = ", ".join(required)
                lines.append(f"• {name}({params_str}) — {desc}")
            else:
                props = list(params.get("properties", {}).keys())[:3]
                params_str = ", ".join(props) if props else ""
                lines.append(f"• {name}({params_str}) — {desc}")

        return "\n".join(lines)

    # ── Response Parsing ──────────────────────────────────

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
            return {"type": "text", "tool_calls": None, "content": raw}

        # FC mode
        parsed = JSONExtractor.extract(raw)

        if parsed:
            if "content" in parsed and "tool_calls" not in parsed:
                return {
                    "type": "text",
                    "tool_calls": None,
                    "content": str(parsed["content"]),
                }

            tool_calls = self._normalize_tool_calls(parsed, tools)
            if tool_calls:
                names = [tc["function"]["name"] for tc in tool_calls]
                self.logger.info(f"✅ Extracted: {', '.join(names)}")
                return {
                    "type": "tool_calls",
                    "tool_calls": tool_calls,
                    "content": None,
                }

        self.logger.warning("❌ No tool_calls found in response")
        return {"type": "text", "tool_calls": None, "content": raw}

    def _normalize_tool_calls(
        self,
        parsed: Dict[str, Any],
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tool_names = set()
        for t in tools:
            func = t.get("function", {})
            if "name" in func:
                tool_names.add(func["name"])

        raw_calls = []

        if "tool_calls" in parsed:
            tc = parsed["tool_calls"]
            raw_calls = tc if isinstance(tc, list) else [tc]
        elif "function_call" in parsed:
            raw_calls = [parsed["function_call"]]
        elif "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
            raw_calls = [parsed]
        elif "function" in parsed and isinstance(parsed["function"], dict):
            raw_calls = [parsed]

        if not raw_calls:
            return []

        result = []
        for raw in raw_calls:
            name = None
            arguments = {}

            if "function" in raw and isinstance(raw["function"], dict):
                name = raw["function"].get("name")
                arguments = raw["function"].get("arguments", {})
            else:
                name = raw.get("name", raw.get("function_name"))
                arguments = raw.get(
                    "arguments",
                    raw.get("parameters", raw.get("params", {}))
                )

            if not name:
                continue

            matched = self._match_name(name, tool_names)
            if not matched:
                continue

            if isinstance(arguments, dict):
                args_str = json.dumps(arguments, ensure_ascii=False)
            elif isinstance(arguments, str):
                try:
                    json.loads(arguments)
                    args_str = arguments
                except (json.JSONDecodeError, ValueError):
                    args_str = json.dumps({"input": arguments})
            else:
                args_str = "{}"

            result.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": matched, "arguments": args_str},
            })

        return result

    @staticmethod
    def _match_name(candidate: str, available: set) -> Optional[str]:
        if candidate in available:
            return candidate

        cl = candidate.lower().strip()
        for name in available:
            if name.lower() == cl:
                return name

        for name in available:
            if cl in name.lower() or name.lower() in cl:
                return name

        cn = cl.replace(" ", "_").replace("-", "_")
        for name in available:
            if name.lower().replace(" ", "_").replace("-", "_") == cn:
                return name

        return None

    # ── Response Building (PicoClaw Compatible) ───────────

    def build_response(
        self,
        parsed: Dict[str, Any],
        model: str,
        prompt_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Build OpenAI-format response (PicoClaw compatible)."""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if parsed["type"] == "tool_calls" and parsed["tool_calls"]:
            message = {
                "role": "assistant",
                "content": "",  # PicoClaw expects string, not null
                "tool_calls": parsed["tool_calls"],
            }
            finish_reason = "tool_calls"
            comp_tokens = len(json.dumps(parsed["tool_calls"])) // 4
        else:
            content = parsed.get("content") or ""
            message = {
                "role": "assistant",
                "content": content,
            }
            finish_reason = "stop"
            comp_tokens = max(1, len(content) // 4)

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens": prompt_tokens + comp_tokens,
            },
        }


# Global
_emulator: Optional[FunctionCallingEmulator] = None


def get_emulator() -> FunctionCallingEmulator:
    global _emulator
    if _emulator is None:
        _emulator = FunctionCallingEmulator()
    return _emulator