"""
Function Calling Emulator v4 — Robust parsing + strict normalization.
Fixes:
- Stronger prompt with exact format example
- Better JSON extraction
- Normalize non-standard formats (tool → function.name)
- Validate before sending to PicoClaw
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
        for pattern in [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]:
            match = re.search(pattern, text)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue

        # Strategy 3: Find first complete {...} with bracket matching
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

        # Prefer candidates with tool_calls or content
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
    """
    FC Emulator v4 — Strict format enforcement.

    Key improvements:
    - Exact format example in prompt
    - Normalize any weird format to OpenAI standard
    - Validate before returning
    - Strip explanation text
    """

    # Strict prompt with EXACT format example
    FC_PROMPT = """You are a function-calling AI. You MUST respond with ONLY a valid JSON object.

AVAILABLE FUNCTIONS:
{tools}

OUTPUT FORMAT (copy this EXACT structure):
{{"tool_calls":[{{"function":{{"name":"FUNCTION_NAME","arguments":{{"param":"value"}}}}}}]}}

EXAMPLES:
User: "What's the weather in Tokyo?"
Output: {{"tool_calls":[{{"function":{{"name":"get_weather","arguments":{{"location":"Tokyo"}}}}}}]}}

User: "Search for Python tutorials"
Output: {{"tool_calls":[{{"function":{{"name":"web_search","arguments":{{"query":"Python tutorials"}}}}}}]}}

User: "How much free RAM?"
Output: {{"tool_calls":[{{"function":{{"name":"get_free_ram","arguments":{{}}}}}}]}}

CRITICAL RULES:
1. Output ONLY the JSON object - NO text before or after
2. NO markdown, NO backticks, NO explanation
3. Use "function" with "name" and "arguments" - NOT "tool" or other keys
4. "arguments" must be an object {{}} even if empty
5. Pick the MOST relevant function from the list"""

    FC_RETRY_PROMPT = """OUTPUT ONLY THIS JSON (fill in the values):
{{"tool_calls":[{{"function":{{"name":"{fname}","arguments":{args_hint}}}}}]}}

NO other text. ONLY the JSON above."""

    def __init__(self) -> None:
        self.logger = get_logger("function_calling")

    # ══════════════════════════════════════════════════════
    # Detection
    # ══════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════
    # Message Building
    # ══════════════════════════════════════════════════════

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

            tool_str = self._format_tools_for_prompt(tools)
            fc_prompt = self.FC_PROMPT.format(tools=tool_str)

            if tool_choice == "required":
                fc_prompt += "\n\nYou MUST call a function. Text response is FORBIDDEN."
            elif isinstance(tool_choice, dict):
                fname = tool_choice.get("function", {}).get("name", "")
                if fname:
                    fc_prompt += f"\n\nYou MUST call the function: {fname}"

            enhanced = [{"role": "system", "content": fc_prompt}]

            for msg in messages:
                if msg.get("role") == "system":
                    original = msg.get("content", "")
                    if original and len(original) < 300:
                        enhanced[0]["content"] += f"\n\nAdditional context: {original}"
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
        """Build retry with specific function hint."""
        best_func = self._guess_best_function(user_message, tools)

        if best_func:
            func = best_func.get("function", {})
            fname = func.get("name", "unknown")
            params = func.get("parameters", {})
            props = params.get("properties", {})

            args_hint = {}
            for pname in props.keys():
                args_hint[pname] = "..."

            if not args_hint:
                args_hint_str = "{}"
            else:
                args_hint_str = json.dumps(args_hint)

            retry_prompt = self.FC_RETRY_PROMPT.format(
                fname=fname,
                args_hint=args_hint_str,
            )
        else:
            retry_prompt = (
                'Output ONLY: {"tool_calls":[{"function":{"name":"...","arguments":{...}}}]}'
            )

        msgs = original_messages.copy()
        msgs.append({"role": "user", "content": retry_prompt})
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
                    "content": (
                        f"[Tool Result for {tool_call_id}]:\n"
                        f"{content}\n\n"
                        f"Now provide your final response based on this result."
                    ),
                })

            elif role == "assistant" and msg.get("tool_calls"):
                calls = msg.get("tool_calls", [])
                call_desc = []
                for tc in calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")
                    call_desc.append(f"{name}({args})")

                cleaned.append({
                    "role": "assistant",
                    "content": f"I called: {', '.join(call_desc)}",
                })

            else:
                cleaned.append(msg.copy())

        return cleaned

    def _format_tools_for_prompt(
        self, tools: List[Dict[str, Any]]
    ) -> str:
        """Format tools clearly for the prompt."""
        lines = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "No description")[:100]
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            param_parts = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")[:30]
                req = "(required)" if pname in required else "(optional)"
                param_parts.append(f"    - {pname}: {ptype} {req} {pdesc}")

            lines.append(f"• {name}: {desc}")
            if param_parts:
                lines.extend(param_parts)

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════
    # Response Parsing
    # ══════════════════════════════════════════════════════

    def parse_response(
        self,
        raw: str,
        tools: List[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, Any]:
        if not raw or not raw.strip():
            return {"type": "text", "tool_calls": None, "content": ""}

        # For result/chat mode — return text
        if mode in ("result", "chat"):
            # Try to extract content from JSON if present
            parsed = JSONExtractor.extract(raw)
            if parsed and "content" in parsed:
                return {
                    "type": "text",
                    "tool_calls": None,
                    "content": str(parsed["content"]),
                }
            # Strip any JSON and return clean text
            clean_text = self._strip_json_from_text(raw)
            return {"type": "text", "tool_calls": None, "content": clean_text}

        # ── FC mode ───────────────────────────────────────
        parsed = JSONExtractor.extract(raw)

        if parsed:
            # Check if it's a text response
            if "content" in parsed and "tool_calls" not in parsed:
                return {
                    "type": "text",
                    "tool_calls": None,
                    "content": str(parsed["content"]),
                }

            # Normalize and validate tool_calls
            tool_calls = self._normalize_and_validate(parsed, tools)
            if tool_calls:
                names = [tc["function"]["name"] for tc in tool_calls]
                self.logger.info(f"✅ Valid tool_calls: {', '.join(names)}")
                return {
                    "type": "tool_calls",
                    "tool_calls": tool_calls,
                    "content": None,
                }

        # Try text intent extraction as fallback
        tool_calls = self._extract_intent_from_text(raw, tools)
        if tool_calls:
            names = [tc["function"]["name"] for tc in tool_calls]
            self.logger.info(f"✅ Extracted from text: {', '.join(names)}")
            return {
                "type": "tool_calls",
                "tool_calls": tool_calls,
                "content": None,
            }

        self.logger.warning(
            f"❌ No valid tool_calls. Raw: {raw[:80].replace(chr(10), ' ')}..."
        )
        return {"type": "text", "tool_calls": None, "content": raw}

    # ══════════════════════════════════════════════════════
    # Normalization & Validation (KEY FIX)
    # ══════════════════════════════════════════════════════

    def _normalize_and_validate(
        self,
        parsed: Dict[str, Any],
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalize ANY format to OpenAI standard and validate.

        Handles:
        - {"tool_calls":[{"tool":"x","amount":"y"}]} → normalize
        - {"tool_calls":[{"name":"x","arguments":{}}]} → normalize
        - {"tool_calls":[{"function":{"name":"x"}}]} → validate
        - {"function_call":{"name":"x"}} → convert
        """
        tool_names = {
            t.get("function", {}).get("name", "").lower(): t.get("function", {}).get("name", "")
            for t in tools
        }

        raw_calls = []

        # Extract raw calls from various formats
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

        # Normalize each call
        result = []
        for raw in raw_calls:
            if not isinstance(raw, dict) or not raw:
                continue

            normalized = self._normalize_single_call(raw, tool_names)
            if normalized:
                result.append(normalized)

        return result

    def _normalize_single_call(
        self,
        raw: Dict[str, Any],
        tool_names: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize a single tool call to OpenAI format.

        Input formats handled:
        - {"function":{"name":"x","arguments":{}}}  ← OpenAI standard
        - {"name":"x","arguments":{}}               ← Flat format
        - {"tool":"x","param":"value"}              ← Wrong format (normalize)
        - {"function_name":"x","params":{}}         ← Alternative format
        """
        name = None
        arguments = {}

        # ── Extract name ──────────────────────────────────
        if "function" in raw and isinstance(raw["function"], dict):
            name = raw["function"].get("name")
            arguments = raw["function"].get("arguments", {})
        elif "name" in raw:
            name = raw["name"]
            arguments = raw.get("arguments", raw.get("parameters", raw.get("params", {})))
        elif "tool" in raw:
            # Wrong format: {"tool":"get_free_ram","amount":"?"}
            name = raw["tool"]
            # Collect remaining keys as arguments
            arguments = {
                k: v for k, v in raw.items()
                if k not in ("tool", "id", "type", "index")
            }
        elif "function_name" in raw:
            name = raw["function_name"]
            arguments = raw.get("arguments", raw.get("parameters", raw.get("params", {})))

        if not name:
            return None

        # ── Match name to available tools ─────────────────
        name_lower = str(name).lower().strip()
        matched_name = None

        # Exact match
        if name_lower in tool_names:
            matched_name = tool_names[name_lower]
        else:
            # Fuzzy match
            for tn_lower, tn_original in tool_names.items():
                if name_lower in tn_lower or tn_lower in name_lower:
                    matched_name = tn_original
                    break
                # Underscore/hyphen normalization
                norm_input = name_lower.replace("-", "_").replace(" ", "_")
                norm_tool = tn_lower.replace("-", "_").replace(" ", "_")
                if norm_input == norm_tool:
                    matched_name = tn_original
                    break

        if not matched_name:
            self.logger.warning(f"Unknown function: {name}")
            return None

        # ── Normalize arguments ───────────────────────────
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                # If it's just a string, wrap it
                arguments = {"input": arguments}

        if not isinstance(arguments, dict):
            arguments = {}

        # Clean up weird argument values
        cleaned_args = {}
        for k, v in arguments.items():
            # Skip placeholder values
            if v in ("?", "...", "<value>", "VALUE", None):
                continue
            cleaned_args[k] = v

        # Convert arguments to JSON string (OpenAI format)
        args_str = json.dumps(cleaned_args, ensure_ascii=False)

        return {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": matched_name,
                "arguments": args_str,
            },
        }

    # ══════════════════════════════════════════════════════
    # Text Intent Extraction (Fallback)
    # ══════════════════════════════════════════════════════

    def _extract_intent_from_text(
        self,
        text: str,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract function call intent from natural language."""
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
            name_words = fname_lower.replace("_", " ").split()

            matched = False

            # Direct mention
            if fname_lower in text_lower:
                matched = True
            # All name words present
            elif all(w in text_lower for w in name_words if len(w) > 2):
                matched = True
            # Description keywords
            else:
                keywords = [
                    w for w in fdesc.split()
                    if len(w) > 3 and w not in (
                        "the", "for", "and", "this", "that", "with", "from"
                    )
                ]
                if keywords:
                    match_count = sum(1 for kw in keywords if kw in text_lower)
                    if match_count >= max(1, len(keywords) // 2):
                        matched = True

            if not matched:
                continue

            # Extract arguments
            arguments = {}
            for param_name in props.keys():
                value = self._extract_param_value(text, param_name, props[param_name])
                if value:
                    arguments[param_name] = value

            # Use full text as first param if no args extracted
            if required and not arguments:
                clean = re.sub(
                    r'^(please |can you |could you |i want to |'
                    r'i need to |let me |search for |find |'
                    r'look up |get |what is |what\'s |how much )',
                    '', text.strip(), flags=re.IGNORECASE,
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

    def _extract_param_value(
        self,
        text: str,
        param_name: str,
        param_info: Dict[str, Any],
    ) -> Optional[str]:
        """Extract parameter value from text."""
        patterns = {
            "query": [r'(?:search|find|look up)\s+(?:for\s+)?["\']?(.+?)["\']?(?:\.|$)'],
            "location": [r'(?:in|at|for)\s+([A-Z][a-zA-Z\s,]+?)(?:\.|$|\?)'],
            "url": [r'(https?://\S+)'],
            "city": [r'(?:in|at|for)\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$|\?)'],
            "unit": [r'\b(MB|GB|KB|bytes?)\b'],
        }

        name_lower = param_name.lower()
        desc_lower = param_info.get("description", "").lower()

        for key, pats in patterns.items():
            if key in name_lower or key in desc_lower:
                for pat in pats:
                    match = re.search(pat, text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()

        # Quoted strings
        quoted = re.findall(r'["\'](.+?)["\']', text)
        if quoted:
            return quoted[0]

        return None

    def _guess_best_function(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Guess which function matches user intent."""
        if not tools:
            return None

        msg_lower = user_message.lower()
        best_tool = None
        best_score = 0

        for tool in tools:
            func = tool.get("function", {})
            fname = func.get("name", "").lower()
            fdesc = func.get("description", "").lower()

            score = 0

            for w in fname.replace("_", " ").split():
                if len(w) > 2 and w in msg_lower:
                    score += 3

            for w in fdesc.split():
                if len(w) > 3 and w in msg_lower:
                    score += 1

            if score > best_score:
                best_score = score
                best_tool = tool

        return best_tool or tools[0]

    def _strip_json_from_text(self, text: str) -> str:
        """Remove JSON objects from text, keep only prose."""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)

        # Remove JSON objects
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

    # ══════════════════════════════════════════════════════
    # Response Building
    # ══════════════════════════════════════════════════════

    def build_response(
        self,
        parsed: Dict[str, Any],
        model: str,
        prompt_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Build OpenAI-format response."""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if parsed["type"] == "tool_calls" and parsed["tool_calls"]:
            # Ensure all tool_calls are properly formatted
            valid_calls = []
            for tc in parsed["tool_calls"]:
                if self._validate_tool_call(tc):
                    valid_calls.append(tc)

            if valid_calls:
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": valid_calls,
                }
                finish_reason = "tool_calls"
                comp_tokens = len(json.dumps(valid_calls)) // 4
            else:
                # Fallback to text if no valid calls
                message = {
                    "role": "assistant",
                    "content": "I couldn't process that request properly.",
                }
                finish_reason = "stop"
                comp_tokens = 10
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

    def _validate_tool_call(self, tc: Dict[str, Any]) -> bool:
        """Validate a tool call has correct structure."""
        if not isinstance(tc, dict):
            return False
        if "function" not in tc:
            return False
        func = tc["function"]
        if not isinstance(func, dict):
            return False
        if "name" not in func or not func["name"]:
            return False
        if "arguments" not in func:
            return False
        # Arguments must be a string
        if not isinstance(func["arguments"], str):
            tc["function"]["arguments"] = json.dumps(func["arguments"])
        return True


# ═══════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════

_emulator: Optional[FunctionCallingEmulator] = None


def get_emulator() -> FunctionCallingEmulator:
    global _emulator
    if _emulator is None:
        _emulator = FunctionCallingEmulator()
    return _emulator