"""
g4f Provider FC Tester
Tests ALL providers for Function Calling compatibility.
Ranks them by reliability.

Usage:
  python test_providers.py

Output:
  - Ranking provider terbaik untuk FC
  - Ranking provider terbaik untuk chat
  - Sample response dari masing-masing
"""

import json
import time
import sys
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# Test Configuration
# ═══════════════════════════════════════════════════════════

# FC Test: Minta AI return tool_calls JSON
FC_TEST_MESSAGES = [
    {
        "role": "system",
        "content": (
            'You are a JSON-only assistant. Respond with EXACTLY this format:\n'
            '{"tool_calls":[{"name":"FUNCTION","arguments":{"param":"value"}}]}\n'
            'NO text, NO markdown, NO explanation. ONLY JSON.'
        )
    },
    {
        "role": "user",
        "content": "Search for weather in Tokyo"
    }
]

# Chat Test: Normal chat
CHAT_TEST_MESSAGES = [
    {
        "role": "user",
        "content": "Say exactly: HELLO_TEST_OK"
    }
]

# Tool Result Test: Kirim hasil tool, minta AI reasoning
TOOL_RESULT_TEST_MESSAGES = [
    {
        "role": "user",
        "content": "What is the weather in Tokyo?"
    },
    {
        "role": "assistant",
        "content": "I need to search for that information."
    },
    {
        "role": "user",
        "content": (
            "[Tool Result]: Weather in Tokyo: Sunny, 25°C, humidity 60%.\n\n"
            "Based on this result, give me a brief answer."
        )
    }
]

# Models to test per provider
TEST_MODELS = ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "auto", ""]

TIMEOUT = 30  # seconds per test


# ═══════════════════════════════════════════════════════════
# JSON Checker
# ═══════════════════════════════════════════════════════════

def extract_json(text: str) -> Optional[Dict]:
    """Try to extract JSON from text."""
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Pure JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find {...}
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except (json.JSONDecodeError, ValueError):
                    return None

    return None


def check_fc_quality(response: str) -> Dict[str, Any]:
    """
    Check how well a response matches OpenAI FC format.
    
    Returns quality report.
    """
    result = {
        "raw_response": response[:500] if response else "",
        "has_json": False,
        "has_tool_calls": False,
        "has_name": False,
        "has_arguments": False,
        "is_pure_json": False,
        "has_wrapper_text": False,
        "format_score": 0,  # 0-100
        "parsed_json": None,
        "issues": []
    }

    if not response:
        result["issues"].append("Empty response")
        return result

    text = response.strip()

    # Check if pure JSON (no text before/after)
    try:
        parsed = json.loads(text)
        result["is_pure_json"] = True
        result["has_json"] = True
        result["parsed_json"] = parsed
        result["format_score"] += 30
    except (json.JSONDecodeError, ValueError):
        # Try extract from mixed text
        parsed = extract_json(text)
        if parsed:
            result["has_json"] = True
            result["has_wrapper_text"] = True
            result["parsed_json"] = parsed
            result["format_score"] += 15
            result["issues"].append("JSON wrapped in text")
        else:
            result["issues"].append("No valid JSON found")
            return result

    if not parsed:
        return result

    # Check for tool_calls
    if "tool_calls" in parsed:
        result["has_tool_calls"] = True
        result["format_score"] += 25

        calls = parsed["tool_calls"]
        if isinstance(calls, list) and calls:
            call = calls[0] if calls else {}

            # Check for name
            name = (
                call.get("name") or
                call.get("tool") or
                call.get("function", {}).get("name") if isinstance(call.get("function"), dict) else None
            )
            if name:
                result["has_name"] = True
                result["format_score"] += 20

            # Check for arguments
            args = (
                call.get("arguments") or
                call.get("params") or
                call.get("parameters") or
                (call.get("function", {}).get("arguments") if isinstance(call.get("function"), dict) else None)
            )
            if args:
                result["has_arguments"] = True
                result["format_score"] += 15

            # Check OpenAI exact format
            if isinstance(call.get("function"), dict):
                if call["function"].get("name") and call["function"].get("arguments"):
                    result["format_score"] += 10  # Bonus: exact OpenAI format!

    elif "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
        # Flat format (not wrapped in tool_calls)
        result["has_name"] = True
        result["has_arguments"] = True
        result["format_score"] += 40
        result["issues"].append("Missing tool_calls wrapper (flat format)")

    elif "function_call" in parsed:
        result["has_tool_calls"] = True
        result["format_score"] += 30
        result["issues"].append("Legacy function_call format")

    elif "content" in parsed and "tool_calls" not in parsed:
        result["format_score"] += 5
        result["issues"].append("Text-only response in JSON")

    else:
        result["issues"].append(f"Unknown JSON structure: {list(parsed.keys())}")

    return result


# ═══════════════════════════════════════════════════════════
# Provider Discovery
# ═══════════════════════════════════════════════════════════

def discover_providers() -> List[Dict[str, Any]]:
    """Discover all working g4f providers."""
    import inspect

    providers = []

    try:
        import g4f
        from g4f import Provider
    except ImportError:
        print("ERROR: g4f not installed!")
        print("Run: pip install g4f")
        sys.exit(1)

    # Skip these
    skip = {
        'BaseProvider', 'AsyncProvider', 'RetryProvider',
        'BaseRetryProvider', 'AsyncGeneratorProvider',
        'ProviderModelMixin', 'AbstractProvider',
        'CreateImagesProvider',
    }

    for name in sorted(dir(Provider)):
        if name.startswith('_') or name in skip:
            continue

        attr = getattr(Provider, name, None)
        if attr is None or not inspect.isclass(attr):
            continue

        working = getattr(attr, 'working', False)
        needs_auth = getattr(attr, 'needs_auth', False)

        if not working or needs_auth:
            continue

        # Get models
        models = []
        for attr_name in ('models', 'model', 'default_model'):
            val = getattr(attr, attr_name, None)
            if isinstance(val, (list, tuple, set)):
                models = list(val)
                break
            elif isinstance(val, str) and val:
                models = [val]
                break

        supports_stream = getattr(attr, 'supports_stream', False)

        providers.append({
            "name": name,
            "class": attr,
            "models": models[:5],  # Limit
            "supports_stream": supports_stream,
            "needs_auth": needs_auth
        })

    return providers


# ═══════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════

def test_provider(
    provider_info: Dict,
    messages: List[Dict],
    test_name: str
) -> Dict[str, Any]:
    """Test a single provider."""
    import g4f

    name = provider_info["name"]
    provider_class = provider_info["class"]
    models = provider_info["models"]

    # Pick model
    model = ""
    for preferred in TEST_MODELS:
        if preferred in models or preferred == "" or preferred == "auto":
            model = preferred
            break
    if not model and models:
        model = models[0]

    result = {
        "provider": name,
        "model": model,
        "test": test_name,
        "success": False,
        "response": "",
        "error": "",
        "time_seconds": 0,
        "fc_quality": None
    }

    try:
        start = time.time()

        response = g4f.ChatCompletion.create(
            model=model,
            messages=messages,
            provider=provider_class,
            timeout=TIMEOUT
        )

        elapsed = time.time() - start
        response_str = str(response).strip()

        result["success"] = bool(response_str)
        result["response"] = response_str[:1000]
        result["time_seconds"] = round(elapsed, 2)

        # FC quality check
        if test_name == "fc":
            result["fc_quality"] = check_fc_quality(response_str)

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def run_all_tests():
    """Run all tests on all providers."""

    print()
    print("=" * 70)
    print("  g4f PROVIDER FUNCTION CALLING TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Discover
    print("🔍 Discovering providers...")
    providers = discover_providers()
    print(f"   Found {len(providers)} working providers\n")

    if not providers:
        print("ERROR: No providers found!")
        return

    # Results storage
    fc_results = []
    chat_results = []
    tool_results = []

    total = len(providers)

    # ── FC Test ─────────────────────────────────────────
    print("=" * 70)
    print("TEST 1: FUNCTION CALLING (Most Important)")
    print("=" * 70)
    print()

    for i, provider in enumerate(providers, 1):
        name = provider["name"]
        print(f"  [{i}/{total}] Testing {name}...", end=" ", flush=True)

        result = test_provider(provider, FC_TEST_MESSAGES, "fc")

        if result["success"]:
            score = result["fc_quality"]["format_score"] if result["fc_quality"] else 0
            has_tc = result["fc_quality"]["has_tool_calls"] if result["fc_quality"] else False

            if score >= 70:
                print(f"✅ Score: {score}/100 (EXCELLENT)")
            elif score >= 40:
                print(f"🟡 Score: {score}/100 (OK)")
            elif score > 0:
                print(f"🟠 Score: {score}/100 (Poor)")
            else:
                print(f"❌ Score: 0 (No JSON)")

            # Show brief response
            resp_preview = result["response"][:80].replace('\n', ' ')
            print(f"         Response: {resp_preview}...")

            if result["fc_quality"] and result["fc_quality"]["issues"]:
                print(f"         Issues: {', '.join(result['fc_quality']['issues'])}")
        else:
            print(f"💀 FAILED: {result['error'][:60]}")

        fc_results.append(result)
        print()

    # ── Chat Test ───────────────────────────────────────
    print()
    print("=" * 70)
    print("TEST 2: NORMAL CHAT")
    print("=" * 70)
    print()

    for i, provider in enumerate(providers, 1):
        name = provider["name"]
        print(f"  [{i}/{total}] Testing {name}...", end=" ", flush=True)

        result = test_provider(provider, CHAT_TEST_MESSAGES, "chat")

        if result["success"]:
            has_keyword = "HELLO_TEST_OK" in result["response"]
            speed = result["time_seconds"]
            print(f"✅ {speed}s {'(exact match)' if has_keyword else '(response OK)'}")
        else:
            print(f"💀 FAILED: {result['error'][:60]}")

        chat_results.append(result)

    # ── Tool Result Test ────────────────────────────────
    print()
    print("=" * 70)
    print("TEST 3: TOOL RESULT PROCESSING")
    print("=" * 70)
    print()

    for i, provider in enumerate(providers, 1):
        name = provider["name"]
        print(f"  [{i}/{total}] Testing {name}...", end=" ", flush=True)

        result = test_provider(provider, TOOL_RESULT_TEST_MESSAGES, "tool_result")

        if result["success"]:
            has_weather = any(
                w in result["response"].lower()
                for w in ["sunny", "25", "tokyo", "weather", "cerah"]
            )
            speed = result["time_seconds"]
            print(f"✅ {speed}s {'(understood result)' if has_weather else '(response OK)'}")
        else:
            print(f"💀 FAILED: {result['error'][:60]}")

        tool_results.append(result)

    # ═══════════════════════════════════════════════════════
    # RANKING
    # ═══════════════════════════════════════════════════════

    print()
    print()
    print("=" * 70)
    print("  📊 FINAL RANKING")
    print("=" * 70)

    # ── FC Ranking ──────────────────────────────────────
    print()
    print("🏆 TOP PROVIDERS FOR FUNCTION CALLING:")
    print("-" * 70)

    fc_scored = []
    for r in fc_results:
        if r["success"] and r["fc_quality"]:
            score = r["fc_quality"]["format_score"]
            fc_scored.append({
                "provider": r["provider"],
                "score": score,
                "time": r["time_seconds"],
                "pure_json": r["fc_quality"]["is_pure_json"],
                "has_tool_calls": r["fc_quality"]["has_tool_calls"],
                "has_name": r["fc_quality"]["has_name"],
                "has_args": r["fc_quality"]["has_arguments"],
                "issues": r["fc_quality"]["issues"],
                "response": r["response"][:200]
            })

    fc_scored.sort(key=lambda x: (-x["score"], x["time"]))

    if fc_scored:
        for i, entry in enumerate(fc_scored[:15], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            flags = []
            if entry["pure_json"]:
                flags.append("PURE_JSON")
            if entry["has_tool_calls"]:
                flags.append("TOOL_CALLS")
            if entry["has_name"]:
                flags.append("NAME")
            if entry["has_args"]:
                flags.append("ARGS")

            print(
                f"  {emoji} {entry['provider']:20s} "
                f"Score: {entry['score']:3d}/100  "
                f"Time: {entry['time']:5.1f}s  "
                f"[{', '.join(flags)}]"
            )
            if entry["issues"]:
                print(f"     Issues: {', '.join(entry['issues'])}")
    else:
        print("  ❌ No providers passed FC test!")

    # ── Chat Ranking ────────────────────────────────────
    print()
    print("💬 TOP PROVIDERS FOR CHAT:")
    print("-" * 70)

    chat_scored = [
        r for r in chat_results if r["success"]
    ]
    chat_scored.sort(key=lambda x: x["time_seconds"])

    for i, entry in enumerate(chat_scored[:10], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        print(
            f"  {emoji} {entry['provider']:20s} "
            f"Time: {entry['time_seconds']:5.1f}s"
        )

    # ── Tool Result Ranking ─────────────────────────────
    print()
    print("🔧 TOP PROVIDERS FOR TOOL RESULT PROCESSING:")
    print("-" * 70)

    tool_scored = [
        r for r in tool_results if r["success"]
    ]
    tool_scored.sort(key=lambda x: x["time_seconds"])

    for i, entry in enumerate(tool_scored[:10], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        understood = any(
            w in entry["response"].lower()
            for w in ["sunny", "25", "tokyo", "weather"]
        )
        print(
            f"  {emoji} {entry['provider']:20s} "
            f"Time: {entry['time_seconds']:5.1f}s  "
            f"{'✅ Understood' if understood else '⚠️ Generic'}"
        )

    # ── Combined Score ──────────────────────────────────
    print()
    print("⭐ OVERALL BEST PROVIDERS (Combined Score):")
    print("-" * 70)

    # Combine all scores
    combined = {}
    for r in fc_results:
        name = r["provider"]
        if name not in combined:
            combined[name] = {"fc": 0, "chat": 0, "tool": 0, "total_time": 0, "tests_passed": 0}
        if r["success"] and r["fc_quality"]:
            combined[name]["fc"] = r["fc_quality"]["format_score"]
            combined[name]["total_time"] += r["time_seconds"]
            combined[name]["tests_passed"] += 1

    for r in chat_results:
        name = r["provider"]
        if name not in combined:
            combined[name] = {"fc": 0, "chat": 0, "tool": 0, "total_time": 0, "tests_passed": 0}
        if r["success"]:
            combined[name]["chat"] = 50  # Base chat score
            combined[name]["total_time"] += r["time_seconds"]
            combined[name]["tests_passed"] += 1

    for r in tool_results:
        name = r["provider"]
        if name not in combined:
            combined[name] = {"fc": 0, "chat": 0, "tool": 0, "total_time": 0, "tests_passed": 0}
        if r["success"]:
            understood = any(
                w in r["response"].lower()
                for w in ["sunny", "25", "tokyo", "weather"]
            )
            combined[name]["tool"] = 50 if understood else 25
            combined[name]["total_time"] += r["time_seconds"]
            combined[name]["tests_passed"] += 1

    # Calculate total
    ranked = []
    for name, scores in combined.items():
        total = scores["fc"] + scores["chat"] + scores["tool"]
        avg_time = scores["total_time"] / max(scores["tests_passed"], 1)
        ranked.append({
            "provider": name,
            "total_score": total,
            "fc_score": scores["fc"],
            "chat_score": scores["chat"],
            "tool_score": scores["tool"],
            "avg_time": avg_time,
            "tests_passed": scores["tests_passed"]
        })

    ranked.sort(key=lambda x: (-x["total_score"], x["avg_time"]))

    for i, entry in enumerate(ranked[:10], 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
        print(
            f"  {emoji} {entry['provider']:20s} "
            f"Total: {entry['total_score']:3d}/200  "
            f"FC: {entry['fc_score']:3d}  "
            f"Chat: {entry['chat_score']:2d}  "
            f"Tool: {entry['tool_score']:2d}  "
            f"Speed: {entry['avg_time']:5.1f}s  "
            f"({entry['tests_passed']}/3 tests)"
        )

    # ── Save Results ────────────────────────────────────
    print()
    print("=" * 70)

    # Save to file
    output_dir = Path.home() / "g4f-bridge" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "provider_test_results.json"

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "total_providers": len(providers),
        "fc_results": [
            {
                "provider": r["provider"],
                "success": r["success"],
                "score": r["fc_quality"]["format_score"] if r.get("fc_quality") else 0,
                "time": r["time_seconds"],
                "response_preview": r["response"][:300],
                "error": r["error"]
            }
            for r in fc_results
        ],
        "ranking": ranked[:10],
        "recommendation": ranked[0]["provider"] if ranked else "none"
    }

    try:
        with open(output_file, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"📁 Results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

    # ── Recommendation ──────────────────────────────────
    print()
    if ranked:
        best = ranked[0]
        print("=" * 70)
        print(f"  🎯 RECOMMENDATION")
        print("=" * 70)
        print()
        print(f"  Best provider for PicoClaw: {best['provider']}")
        print(f"  FC Score: {best['fc_score']}/100")
        print(f"  Average Speed: {best['avg_time']:.1f}s")
        print()
        print(f"  Add to config.json:")
        print(f'  "provider_lock": {{')
        print(f'    "strict_provider_mode": true,')
        print(f'    "locked_provider": "{best["provider"]}",')
        print(f'    "locked_model": "",')
        print(f'    "fail_on_lock_error": false')
        print(f'  }}')
        print()

        if best["fc_score"] < 50:
            print("  ⚠️  WARNING: Even the best provider has low FC score!")
            print("  Consider using Groq free API for FC requests.")
            print("  Setup: https://console.groq.com/keys (free, 5 min)")
    else:
        print("  ❌ No working providers found!")
        print("  Try: pip install --upgrade g4f")

    print()
    print("=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    run_all_tests()