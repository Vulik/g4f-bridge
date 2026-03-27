"""
Function Calling Emulator for g4f-Bridge.

Emulates OpenAI Function Calling for free providers that don't
support it natively. Uses prompt engineering + regex parsing to
convert text responses into structured tool_calls.

Flow:
  1. Detect tools in request
  2. Inject hidden system prompt forcing JSON output
  3. Send to g4f provider
  4. Parse raw text response → extract JSON
  5. Normalize to OpenAI tool_calls format
  6. Return to PicoClaw
"""

import re
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from logger_setup import get_logger


class JSONExtractor:
    """
    Extracts JSON objects from mixed text responses.

    Handles multiple formats:
    - Pure JSON
    - JSON in markdown code blocks (```json ... ```)
    - JSON mixed with explanatory text
    - Multiple JSON objects in one response
    """

    @staticmethod
    def extract_all(text: str) -> List[Dict[str, Any]]:
        """
        Extract all valid JSON objects from text.
        Tries multiple strategies in order of reliability.

        Args:
            text: Raw text response from AI

        Returns:
            List of parsed JSON objects
        """
        if not text or not text.strip():
            return []

        results = []

        # Strategy 1: Entire text is valid JSON
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: JSON in markdown code blocks
        code_block_results = JSONExtractor._from_code_blocks(text)
        if code_block_results:
            return code_block_results

        # Strategy 3: Bracket-matching extraction
        bracket_results = JSONExtractor._by_bracket_matching(text)
        if bracket_results:
            return bracket_results

        return results

    @staticmethod
    def _from_code_blocks(text: str) -> List[Dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        results = []

        # Match ```json ... ``` or ``` ... ```
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    if isinstance(parsed, dict):
                        results.append(parsed)
                    elif isinstance(parsed, list):
                        results.extend(
                            item for item in parsed
                            if isinstance(item, dict)
                        )
                except (json.JSONDecodeError, ValueError):
                    continue

            if results:
                return results

        return results

    @staticmethod
    def _by_bracket_matching(text: str) -> List[Dict[str, Any]]:
        """Extract JSON objects using bracket depth tracking."""
        results = []
        i = 0
        n = len(text)

        while i < n:
            if text[i] == '{':
                depth = 0
                start = i
                in_string = False
                escape_next = False

                while i < n:
                    char = text[i]

                    if escape_next:
                        escape_next = False
                        i += 1
                        continue

                    if char == '\\' and in_string:
                        escape_next = True
                        i += 1
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string

                    if not in_string:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                candidate = text[start:i + 1]
                                try:
                                    parsed = json.loads(candidate)
                                    if isinstance(parsed, dict):
                                        results.append(parsed)
                                except (json.JSONDecodeError, ValueError):
                                    pass
                                break

                    i += 1
            i += 1

        return results

    @staticmethod
    def extract_best(text: str, prefer_key: str = "tool_calls") -> Optional[Dict[str, Any]]:
        """
        Extract the best JSON object, preferring one with a specific key.

        Args:
            text: Raw text
            prefer_key: Preferred key to look for

        Returns:
            Best matching JSON object, or None
        """
        all_json = JSONExtractor.extract_all(text)

        if not all_json:
            return None

        # Prefer objects with the target key
        for obj in all_json:
            if prefer_key in obj:
                return obj

        # Prefer objects with 'name' (likely a tool call)
        for obj in all_json:
            if "name" in obj and "arguments" in obj:
                return obj

        # Prefer objects with 'function' key
        for obj in all_json:
            if "function" in obj or "function_call" in obj:
                return obj

        # Return first object
        return all_json[0] if all_json else None


class ToolCallNormalizer:
    """
    Normalizes various tool call formats to OpenAI standard.

    Handles:
    - {"action":"tool_call","tool_calls":[...]}  (our prompted format)
    - {"tool_calls":[{"function":{"name":"...", "arguments":"..."}}]}  (OpenAI)
    - {"name":"...", "arguments":{...}}  (simple format)
    - {"function_call":{"name":"...", "arguments":"..."}}  (legacy OpenAI)
    - [{"name":"...", "arguments":{...}}]  (array format)
    """

    @staticmethod
    def normalize(
        parsed: Dict[str, Any],
        available_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalize parsed JSON to OpenAI tool_calls format.

        Returns:
            List of normalized tool_call objects:
            [
                {
                    "id": "call_xxxxx",
                    "type": "function",
                    "function": {
                        "name": "function_name",
                        "arguments": "{\"param\": \"value\"}"  # STRING
                    }
                }
            ]
        """
        tool_names = {
            t["function"]["name"]
            for t in available_tools
            if "function" in t and "name" in t["function"]
        }

        raw_calls = ToolCallNormalizer._extract_raw_calls(parsed)

        normalized = []
        for raw in raw_calls:
            name, arguments = ToolCallNormalizer._extract_name_args(raw)

            if not name:
                continue

            # Fuzzy match tool name
            matched_name = ToolCallNormalizer._match_tool_name(
                name, tool_names
            )
            if not matched_name:
                continue

            # Ensure arguments is a JSON string
            if isinstance(arguments, dict):
                args_str = json.dumps(arguments, ensure_ascii=False)
            elif isinstance(arguments, str):
                # Validate it's valid JSON
                try:
                    json.loads(arguments)
                    args_str = arguments
                except (json.JSONDecodeError, ValueError):
                    args_str = json.dumps({"input": arguments})
            else:
                args_str = "{}"

            normalized.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": matched_name,
                    "arguments": args_str,
                },
            })

        return normalized

    @staticmethod
    def _extract_raw_calls(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract raw tool call objects from various formats."""
        # Format: {"action":"tool_call", "tool_calls":[...]}
        if "tool_calls" in parsed:
            calls = parsed["tool_calls"]
            if isinstance(calls, list):
                return calls
            if isinstance(calls, dict):
                return [calls]

        # Format: {"function_call": {...}}
        if "function_call" in parsed:
            return [parsed["function_call"]]

        # Format: {"name":"...", "arguments":{...}} (direct tool call)
        if "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
            return [parsed]

        # Format: {"function": {"name":"...", "arguments":"..."}}
        if "function" in parsed and isinstance(parsed["function"], dict):
            return [parsed]

        return []

    @staticmethod
    def _extract_name_args(raw: Dict[str, Any]) -> Tuple[Optional[str], Any]:
        """Extract function name and arguments from a raw call."""
        name = None
        arguments = {}

        # Nested: {"function": {"name": "...", "arguments": ...}}
        if "function" in raw and isinstance(raw["function"], dict):
            func = raw["function"]
            name = func.get("name")
            arguments = func.get("arguments", func.get("parameters", {}))
        else:
            # Flat: {"name": "...", "arguments": ...}
            name = raw.get("name", raw.get("function_name"))
            arguments = raw.get(
                "arguments",
                raw.get("parameters", raw.get("params", {})),
            )

        return name, arguments

    @staticmethod
    def _match_tool_name(
        candidate: str, available: set
    ) -> Optional[str]:
        """Fuzzy match tool name against available tools."""
        if not candidate:
            return None

        # Exact match
        if candidate in available:
            return candidate

        # Case-insensitive match
        candidate_lower = candidate.lower().strip()
        for tool_name in available:
            if tool_name.lower() == candidate_lower:
                return tool_name

        # Partial match (e.g., "search" matches "web_search")
        for tool_name in available:
            if candidate_lower in tool_name.lower():
                return tool_name
            if tool_name.lower() in candidate_lower:
                return tool_name

        # Replace spaces/hyphens with underscores
        normalized = candidate_lower.replace(" ", "_").replace("-", "_")
        for tool_name in available:
            if tool_name.lower().replace(" ", "_").replace("-", "_") == normalized:
                return tool_name

        return None


class FunctionCallingEmulator:
    """
    Main emulator class.

    Handles the full lifecycle:
    1. Detect tools in request
    2. Build enhanced system prompt
    3. Parse response
    4. Format as OpenAI tool_calls
    """

    # System prompt template for forcing JSON output
    SYSTEM_PROMPT_TEMPLATE = """You are a function-calling AI assistant. You MUST respond ONLY with valid JSON.

## Available Functions:
{tool_definitions}

## Response Format:

When you need to call a function, respond with EXACTLY this JSON structure:
{{"tool_calls": [{{"name": "<function_name>", "arguments": {{<parameters>}}}}]}}

When NO function is needed (normal conversation), respond with:
{{"content": "<your text response>"}}

## Examples:

User: "Search for weather in Tokyo"
{{"tool_calls": [{{"name": "web_search", "arguments": {{"query": "weather in Tokyo"}}}}]}}

User: "Hello, how are you?"
{{"content": "Hello! I'm doing well. How can I help you?"}}

User: "Read the file config.json"
{{"tool_calls": [{{"name": "read_file", "arguments": {{"path": "config.json"}}}}]}}

## ABSOLUTE RULES — VIOLATION CAUSES SYSTEM CRASH:
1. Output ONLY the JSON object — NO text before or after
2. NO markdown, NO code blocks, NO backticks, NO explanations
3. Function names must EXACTLY match the list above
4. Include ALL required parameters
5. The JSON must be parseable by json.loads()
{extra_rules}"""

    FORCE_TOOL_RULE = """6. You MUST call at least one function — "content" only response is FORBIDDEN
7. Analyze the user's request and determine which function best serves it"""

    FORCE_SPECIFIC_TOOL_RULE = """6. You MUST call the function: {tool_name}
7. DO NOT use any other function"""

    def __init__(self) -> None:
        """Initialize emulator."""
        self.logger = get_logger("function_calling")
        self.extractor = JSONExtractor()
        self.normalizer = ToolCallNormalizer()

    def has_tools(self, request_body: Dict[str, Any]) -> bool:
        """Check if request contains tools."""
        tools = request_body.get("tools", [])
        tool_choice = request_body.get("tool_choice", "auto")

        # No tools field at all
        if not tools:
            return False

        # tool_choice is explicitly "none"
        if tool_choice == "none":
            return False

        return True

    def build_enhanced_messages(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "auto",
    ) -> List[Dict[str, str]]:
        """
        Build messages with injected system prompt for function calling.

        Args:
            messages: Original messages from PicoClaw
            tools: Tool definitions
            tool_choice: "auto", "required", "none", or specific tool

        Returns:
            Modified messages with injected system prompt
        """
        # Build tool definitions string
        tool_defs = self._format_tool_definitions(tools)

        # Determine extra rules based on tool_choice
        extra_rules = ""
        if tool_choice == "required":
            extra_rules = self.FORCE_TOOL_RULE
        elif isinstance(tool_choice, dict):
            # Specific tool forced
            func_info = tool_choice.get("function", {})
            tool_name = func_info.get("name", "")
            if tool_name:
                extra_rules = self.FORCE_SPECIFIC_TOOL_RULE.format(
                    tool_name=tool_name
                )

        # Build system prompt
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            tool_definitions=tool_defs,
            extra_rules=extra_rules,
        )

        # Clone messages
        enhanced = []

        # Insert FC system prompt as FIRST message
        enhanced.append({
            "role": "system",
            "content": system_prompt,
        })

        # Add original messages (merge existing system prompts)
        for msg in messages:
            if msg.get("role") == "system":
                # Append original system content to our prompt
                enhanced[0]["content"] += (
                    "\n\n## Additional Context:\n" + msg.get("content", "")
                )
            else:
                enhanced.append(msg.copy())

        self.logger.debug(
            f"Injected FC prompt with {len(tools)} tools "
            f"(tool_choice={tool_choice})"
        )

        return enhanced

    def parse_response(
        self,
        raw_response: str,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parse AI response and extract tool_calls or content.

        Args:
            raw_response: Raw text from g4f provider
            tools: Original tool definitions (for validation)

        Returns:
            {
                "type": "tool_calls" | "text" | "error",
                "tool_calls": [...] | None,
                "content": "..." | None,
                "raw": "original response"
            }
        """
        if not raw_response or not raw_response.strip():
            return {
                "type": "error",
                "tool_calls": None,
                "content": "Empty response from provider",
                "raw": raw_response,
            }

        text = raw_response.strip()

        # Try to extract JSON
        parsed = self.extractor.extract_best(text, prefer_key="tool_calls")

        if parsed:
            # Check if it's a text-only response
            if "content" in parsed and "tool_calls" not in parsed:
                content = parsed.get("content", "")
                if isinstance(content, str) and content:
                    return {
                        "type": "text",
                        "tool_calls": None,
                        "content": content,
                        "raw": raw_response,
                    }

            # Check for action field
            action = parsed.get("action", "")
            if action == "text" and "content" in parsed:
                return {
                    "type": "text",
                    "tool_calls": None,
                    "content": parsed["content"],
                    "raw": raw_response,
                }

            # Try to normalize tool_calls
            normalized = self.normalizer.normalize(parsed, tools)

            if normalized:
                self.logger.info(
                    f"Extracted {len(normalized)} tool_call(s): "
                    + ", ".join(tc["function"]["name"] for tc in normalized)
                )
                return {
                    "type": "tool_calls",
                    "tool_calls": normalized,
                    "content": None,
                    "raw": raw_response,
                }

        # Check if multiple JSON objects (multiple tool calls)
        all_json = self.extractor.extract_all(text)
        if all_json:
            all_normalized = []
            for obj in all_json:
                norm = self.normalizer.normalize(obj, tools)
                all_normalized.extend(norm)

            if all_normalized:
                self.logger.info(
                    f"Extracted {len(all_normalized)} tool_call(s) "
                    f"from {len(all_json)} JSON objects"
                )
                return {
                    "type": "tool_calls",
                    "tool_calls": all_normalized,
                    "content": None,
                    "raw": raw_response,
                }

        # Fallback: No valid JSON found → return as text
        self.logger.warning(
            "No valid tool_calls found in response, "
            "returning as text"
        )
        return {
            "type": "text",
            "tool_calls": None,
            "content": text,
            "raw": raw_response,
        }

    def build_openai_response(
        self,
        parsed: Dict[str, Any],
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> Dict[str, Any]:
        """
        Build OpenAI-compatible response.

        Args:
            parsed: Result from parse_response()
            model: Model name
            prompt_tokens: Prompt token count
            completion_tokens: Completion token count

        Returns:
            OpenAI-format response dict
        """
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if parsed["type"] == "tool_calls" and parsed["tool_calls"]:
            # Tool calls response
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": parsed["tool_calls"],
            }
            finish_reason = "tool_calls"
        else:
            # Text response
            content = parsed.get("content") or parsed.get("raw", "")
            message = {
                "role": "assistant",
                "content": content,
            }
            finish_reason = "stop"

        if completion_tokens == 0:
            content_for_count = (
                json.dumps(parsed.get("tool_calls", ""))
                if parsed["type"] == "tool_calls"
                else (parsed.get("content") or "")
            )
            completion_tokens = max(1, len(content_for_count) // 4)

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _format_tool_definitions(
        self, tools: List[Dict[str, Any]]
    ) -> str:
        """Format tool definitions for system prompt."""
        lines = []
        for i, tool in enumerate(tools, 1):
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "No description")
            params = func.get("parameters", {})

            lines.append(f"{i}. **{name}**: {desc}")

            # List parameters
            properties = params.get("properties", {})
            required = params.get("required", [])

            if properties:
                lines.append("   Parameters:")
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    req_marker = " (REQUIRED)" if param_name in required else ""
                    lines.append(
                        f"   - {param_name} ({param_type}){req_marker}: {param_desc}"
                    )

            lines.append("")

        return "\n".join(lines)


# Global instance
_emulator: Optional[FunctionCallingEmulator] = None


def get_emulator() -> FunctionCallingEmulator:
    """Get global emulator instance."""
    global _emulator
    if _emulator is None:
        _emulator = FunctionCallingEmulator()
    return _emulator


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    emulator = FunctionCallingEmulator()

    # Test tools
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # Test 1: Pure JSON response
    print("=== Test 1: Pure JSON ===")
    r1 = emulator.parse_response(
        '{"tool_calls": [{"name": "web_search", "arguments": {"query": "weather"}}]}',
        test_tools,
    )
    print(f"Type: {r1['type']}, Calls: {r1['tool_calls']}")

    # Test 2: JSON in markdown
    print("\n=== Test 2: Markdown JSON ===")
    r2 = emulator.parse_response(
        'Sure! Let me search.\n```json\n{"tool_calls": [{"name": "web_search", "arguments": {"query": "weather"}}]}\n```',
        test_tools,
    )
    print(f"Type: {r2['type']}, Calls: {r2['tool_calls']}")

    # Test 3: Chatty response with embedded JSON
    print("\n=== Test 3: Mixed text + JSON ===")
    r3 = emulator.parse_response(
        'I\'ll search for that. {"name": "web_search", "arguments": {"query": "Tokyo weather"}} Let me get results.',
        test_tools,
    )
    print(f"Type: {r3['type']}, Calls: {r3['tool_calls']}")

    # Test 4: Text-only response
    print("\n=== Test 4: Text only ===")
    r4 = emulator.parse_response(
        '{"content": "Hello! How can I help?"}',
        test_tools,
    )
    print(f"Type: {r4['type']}, Content: {r4['content']}")

    # Test 5: Build OpenAI response
    print("\n=== Test 5: OpenAI response format ===")
    response = emulator.build_openai_response(r1, "gpt-4", 50, 30)
    print(json.dumps(response, indent=2))