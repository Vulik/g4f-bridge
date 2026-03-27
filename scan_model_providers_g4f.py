"""
g4f Full Scan & Test
====================
Step 1: Scan ALL models
Step 2: Scan ALL providers  
Step 3: Map model ↔ provider compatibility
Step 4: Test each combination (FC + Chat)
Step 5: Rank & recommend

Usage:
  python full_scan_test.py              # Full test
  python full_scan_test.py --scan-only  # Scan only (no test)
  python full_scan_test.py --fast       # Test top 20 combinations only
"""

import json
import time
import sys
import re
import os
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict


# ═══════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    providers: List[str] = field(default_factory=list)
    category: str = "unknown"  # chat, reasoning, image, code, etc


@dataclass
class ProviderInfo:
    """Information about a provider."""
    name: str
    models: List[str] = field(default_factory=list)
    working: bool = False
    needs_auth: bool = False
    supports_stream: bool = False
    class_ref: Any = None


@dataclass
class TestResult:
    """Result of a single test."""
    provider: str
    model: str
    test_type: str  # "fc", "chat", "tool_result"
    success: bool = False
    response: str = ""
    error: str = ""
    time_seconds: float = 0.0
    fc_score: int = 0
    is_pure_json: bool = False
    has_tool_calls: bool = False
    has_name: bool = False
    has_arguments: bool = False
    issues: List[str] = field(default_factory=list)


@dataclass
class CombinationScore:
    """Combined score for a provider+model combination."""
    provider: str
    model: str
    fc_score: int = 0
    chat_ok: bool = False
    tool_ok: bool = False
    avg_time: float = 0.0
    total_score: int = 0
    fc_response_preview: str = ""


# ═══════════════════════════════════════════════════════════
# Model Categorizer
# ═══════════════════════════════════════════════════════════

MODEL_CATEGORIES = {
    "flagship": [
        "gpt-4o", "gpt-4", "gpt-5", "gpt-4-turbo",
        "claude-3.5-sonnet", "claude-3-opus", "claude-sonnet-4",
        "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash",
    ],
    "reasoning": [
        "o1", "o3", "o3-mini", "o4-mini",
        "deepseek-r1", "deepseek-reasoner",
        "qwq-32b", "qvq-72b",
    ],
    "code": [
        "deepseek-coder", "codestral", "qwen-2.5-coder",
        "devstral", "gpt-5-codex",
    ],
    "fast": [
        "gpt-3.5-turbo", "gpt-4o-mini",
        "claude-3-haiku", "gemini-1.5-flash",
        "mistral-small", "llama-3.2-3b",
    ],
    "open": [
        "llama-3.3-70b", "llama-3.1-70b",
        "qwen-3-235b", "qwen-2.5-72b",
        "mistral-large", "mixtral-8x22b",
        "gemma-3-27b", "phi-4",
    ],
    "image": [
        "dall-e-3", "flux", "flux-dev", "flux-schnell",
        "stable-diffusion", "sdxl", "midjourney",
    ],
}


def categorize_model(model_name: str) -> str:
    """Categorize a model by name."""
    ml = model_name.lower()
    for category, patterns in MODEL_CATEGORIES.items():
        for pattern in patterns:
            if pattern.lower() in ml:
                return category
    return "other"


# ═══════════════════════════════════════════════════════════
# Scanner
# ═══════════════════════════════════════════════════════════

class G4FScanner:
    """Scans g4f for all models and providers."""

    SKIP_CLASSES = {
        'BaseProvider', 'AsyncProvider', 'RetryProvider',
        'BaseRetryProvider', 'AsyncGeneratorProvider',
        'ProviderModelMixin', 'AbstractProvider',
        'CreateImagesProvider',
    }

    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.providers: Dict[str, ProviderInfo] = {}
        self.compatibility: Dict[str, List[str]] = {}  # model → [providers]

    def scan(self) -> Tuple[int, int]:
        """
        Scan all models and providers.
        
        Returns:
            (num_models, num_providers)
        """
        print("=" * 70)
        print("  STEP 1: SCANNING g4f")
        print("=" * 70)
        print()

        try:
            import g4f
            version = getattr(g4f, '__version__', 'unknown')
            print(f"  g4f version: {version}")
        except ImportError:
            print("  ERROR: g4f not installed!")
            print("  Run: pip install g4f")
            sys.exit(1)

        # Scan providers
        print("\n  🔍 Scanning providers...")
        self._scan_providers()
        print(f"     Found {len(self.providers)} working providers")

        # Scan models (from providers)
        print("\n  🔍 Scanning models...")
        self._scan_models()
        print(f"     Found {len(self.models)} unique models")

        # Try g4f.models if available
        self._scan_g4f_models()

        # Build compatibility map
        print("\n  🔍 Building compatibility map...")
        self._build_compatibility()

        return len(self.models), len(self.providers)

    def _scan_providers(self) -> None:
        """Scan all providers."""
        try:
            from g4f import Provider
        except ImportError:
            return

        for name in sorted(dir(Provider)):
            if name.startswith('_') or name in self.SKIP_CLASSES:
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
            for attr_name in ('models', 'model', 'default_model', 'supported_models'):
                val = getattr(attr, attr_name, None)
                if isinstance(val, (list, tuple)):
                    models = [str(m) for m in val if m]
                    break
                elif isinstance(val, set):
                    models = sorted(str(m) for m in val if m)
                    break
                elif isinstance(val, str) and val:
                    models = [val]
                    break

            self.providers[name] = ProviderInfo(
                name=name,
                models=models,
                working=working,
                needs_auth=needs_auth,
                supports_stream=getattr(attr, 'supports_stream', False),
                class_ref=attr,
            )

    def _scan_models(self) -> None:
        """Extract unique models from providers."""
        for pname, pinfo in self.providers.items():
            for model in pinfo.models:
                if model not in self.models:
                    self.models[model] = ModelInfo(
                        name=model,
                        category=categorize_model(model),
                    )
                self.models[model].providers.append(pname)

    def _scan_g4f_models(self) -> None:
        """Try to scan models from g4f.models module."""
        try:
            from g4f import models as g4f_models
            for name in dir(g4f_models):
                if name.startswith('_'):
                    continue
                attr = getattr(g4f_models, name, None)
                if hasattr(attr, 'name'):
                    model_name = getattr(attr, 'name', '')
                    if model_name and model_name not in self.models:
                        self.models[model_name] = ModelInfo(
                            name=model_name,
                            category=categorize_model(model_name),
                        )
        except Exception:
            pass

    def _build_compatibility(self) -> None:
        """Build model → providers compatibility map."""
        self.compatibility.clear()
        for model_name, model_info in self.models.items():
            if model_info.providers:
                self.compatibility[model_name] = model_info.providers

    def print_summary(self) -> None:
        """Print scan summary."""
        print()
        print("=" * 70)
        print("  SCAN SUMMARY")
        print("=" * 70)

        # Models by category
        categories = {}
        for model in self.models.values():
            cat = model.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(model.name)

        print("\n  📋 MODELS BY CATEGORY:")
        for cat in ["flagship", "reasoning", "code", "fast", "open", "other", "image"]:
            if cat in categories:
                count = len(categories[cat])
                samples = ", ".join(categories[cat][:5])
                if count > 5:
                    samples += f" ... (+{count-5} more)"
                print(f"     {cat:12s}: {count:3d} models  [{samples}]")

        # Providers with most models
        print("\n  📋 TOP PROVIDERS (by model count):")
        sorted_providers = sorted(
            self.providers.values(),
            key=lambda p: len(p.models),
            reverse=True
        )
        for p in sorted_providers[:15]:
            model_samples = ", ".join(p.models[:3])
            if len(p.models) > 3:
                model_samples += f" ... (+{len(p.models)-3})"
            print(f"     {p.name:25s}: {len(p.models):3d} models  [{model_samples}]")

        # Models with most providers
        print("\n  📋 TOP MODELS (by provider count):")
        sorted_models = sorted(
            self.models.values(),
            key=lambda m: len(m.providers),
            reverse=True
        )
        for m in sorted_models[:15]:
            prov_samples = ", ".join(m.providers[:3])
            if len(m.providers) > 3:
                prov_samples += f" ... (+{len(m.providers)-3})"
            print(f"     {m.name:30s}: {len(m.providers):2d} providers  [{prov_samples}]")

        print()


# ═══════════════════════════════════════════════════════════
# Tester
# ═══════════════════════════════════════════════════════════

class G4FTester:
    """Tests provider+model combinations."""

    FC_MESSAGES = [
        {
            "role": "system",
            "content": (
                'Respond ONLY with JSON:\n'
                '{"tool_calls":[{"name":"web_search","arguments":{"query":"VALUE"}}]}\n'
                'NO text. NO markdown. ONLY JSON.'
            )
        },
        {
            "role": "user",
            "content": "Search for weather in Tokyo"
        }
    ]

    CHAT_MESSAGES = [
        {"role": "user", "content": "Say exactly: TEST_OK_123"}
    ]

    TOOL_RESULT_MESSAGES = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me calculate."},
        {
            "role": "user",
            "content": "[Tool Result]: 2+2 = 4\n\nGive the answer in one word."
        }
    ]

    TIMEOUT = 30

    def __init__(self):
        self.results: List[TestResult] = []

    def select_combinations(
        self,
        scanner: G4FScanner,
        mode: str = "full"
    ) -> List[Tuple[str, str]]:
        """
        Select provider+model combinations to test.
        
        Args:
            scanner: Scanner with data
            mode: "full" (all), "fast" (top 20), "smart" (prioritized)
            
        Returns:
            List of (provider_name, model_name) tuples
        """
        combinations = []

        # Get all valid combinations
        for model_name, providers in scanner.compatibility.items():
            # Skip image models
            if scanner.models[model_name].category == "image":
                continue

            for provider_name in providers:
                combinations.append((provider_name, model_name))

        if mode == "full":
            print(f"\n  📊 Testing ALL {len(combinations)} combinations")
            return combinations

        if mode == "fast":
            # Prioritize: flagship > reasoning > code > fast > open > other
            priority = {
                "flagship": 0, "reasoning": 1, "code": 2,
                "fast": 3, "open": 4, "other": 5
            }

            combinations.sort(
                key=lambda x: priority.get(
                    scanner.models.get(x[1], ModelInfo(x[1])).category,
                    99
                )
            )

            limit = min(30, len(combinations))
            print(f"\n  📊 Testing TOP {limit} combinations (fast mode)")
            return combinations[:limit]

        if mode == "smart":
            # Pick 1-2 models per category, test across providers
            selected = set()
            per_category = {}

            for model_name, model_info in scanner.models.items():
                cat = model_info.category
                if cat == "image":
                    continue
                if cat not in per_category:
                    per_category[cat] = []
                per_category[cat].append(model_name)

            # Pick top 2 per category (most providers = most testable)
            target_models = set()
            for cat, models in per_category.items():
                models.sort(key=lambda m: len(scanner.models[m].providers), reverse=True)
                for m in models[:2]:
                    target_models.add(m)

            for pname, model in combinations:
                if model in target_models:
                    selected.add((pname, model))

            result = list(selected)
            print(f"\n  📊 Testing {len(result)} SMART combinations ({len(target_models)} models)")
            return result

        return combinations

    def run_tests(
        self,
        scanner: G4FScanner,
        combinations: List[Tuple[str, str]],
        skip_chat: bool = False,
        skip_tool: bool = False,
    ) -> List[TestResult]:
        """Run all tests."""
        import g4f

        total = len(combinations)
        self.results = []

        print()
        print("=" * 70)
        print("  STEP 2: TESTING COMBINATIONS")
        print("=" * 70)

        for i, (provider_name, model_name) in enumerate(combinations, 1):
            provider_info = scanner.providers.get(provider_name)
            if not provider_info or not provider_info.class_ref:
                continue

            category = scanner.models.get(model_name, ModelInfo(model_name)).category
            
            print(f"\n  [{i}/{total}] {provider_name} + {model_name} ({category})")

            # ── FC Test ───────────────────────────────
            print(f"    FC Test...", end=" ", flush=True)
            fc_result = self._test_single(
                g4f, provider_info, model_name,
                self.FC_MESSAGES, "fc"
            )
            self.results.append(fc_result)

            if fc_result.success:
                score_emoji = "✅" if fc_result.fc_score >= 70 else "🟡" if fc_result.fc_score >= 40 else "🟠" if fc_result.fc_score > 0 else "❌"
                print(f"{score_emoji} Score:{fc_result.fc_score}/100 ({fc_result.time_seconds:.1f}s)")
                if fc_result.fc_score > 0:
                    preview = fc_result.response[:80].replace('\n', ' ')
                    print(f"         → {preview}")
            else:
                error_short = fc_result.error[:50]
                print(f"💀 {error_short}")

            # ── Chat Test ─────────────────────────────
            if not skip_chat:
                print(f"    Chat...", end=" ", flush=True)
                chat_result = self._test_single(
                    g4f, provider_info, model_name,
                    self.CHAT_MESSAGES, "chat"
                )
                self.results.append(chat_result)

                if chat_result.success:
                    has_match = "TEST_OK_123" in chat_result.response
                    print(f"✅ {chat_result.time_seconds:.1f}s {'(exact)' if has_match else ''}")
                else:
                    print(f"💀 {chat_result.error[:50]}")

            # ── Tool Result Test ──────────────────────
            if not skip_tool:
                print(f"    Tool Result...", end=" ", flush=True)
                tool_result = self._test_single(
                    g4f, provider_info, model_name,
                    self.TOOL_RESULT_MESSAGES, "tool_result"
                )
                self.results.append(tool_result)

                if tool_result.success:
                    has_4 = "4" in tool_result.response
                    print(f"✅ {tool_result.time_seconds:.1f}s {'(correct)' if has_4 else ''}")
                else:
                    print(f"💀 {tool_result.error[:50]}")

        return self.results

    def _test_single(
        self,
        g4f_module: Any,
        provider_info: ProviderInfo,
        model: str,
        messages: List[Dict],
        test_type: str,
    ) -> TestResult:
        """Run single test."""
        result = TestResult(
            provider=provider_info.name,
            model=model,
            test_type=test_type,
        )

        try:
            start = time.time()

            response = g4f_module.ChatCompletion.create(
                model=model,
                messages=messages,
                provider=provider_info.class_ref,
                timeout=self.TIMEOUT,
            )

            elapsed = time.time() - start
            response_str = str(response).strip()

            result.success = bool(response_str)
            result.response = response_str[:1000]
            result.time_seconds = round(elapsed, 2)

            # FC quality check
            if test_type == "fc" and response_str:
                self._check_fc_quality(result, response_str)

        except Exception as e:
            result.error = str(e)[:200]

        return result

    def _check_fc_quality(self, result: TestResult, text: str) -> None:
        """Check FC response quality."""
        # Try parse JSON
        parsed = self._extract_json(text)

        if not parsed:
            result.fc_score = 0
            result.issues.append("No JSON found")
            return

        # Check pure JSON
        try:
            json.loads(text.strip())
            result.is_pure_json = True
            result.fc_score += 30
        except (json.JSONDecodeError, ValueError):
            result.fc_score += 10
            result.issues.append("JSON wrapped in text")

        # Check tool_calls
        if "tool_calls" in parsed:
            result.has_tool_calls = True
            result.fc_score += 25

            calls = parsed["tool_calls"]
            if isinstance(calls, list) and calls:
                call = calls[0]

                # Name
                name = (
                    call.get("name") or
                    call.get("tool") or
                    call.get("function_name") or
                    (call.get("function", {}).get("name")
                     if isinstance(call.get("function"), dict) else None)
                )
                if name:
                    result.has_name = True
                    result.fc_score += 20

                # Arguments
                args = (
                    call.get("arguments") or
                    call.get("params") or
                    call.get("parameters") or
                    (call.get("function", {}).get("arguments")
                     if isinstance(call.get("function"), dict) else None)
                )
                if args:
                    result.has_arguments = True
                    result.fc_score += 15

                # Bonus: exact OpenAI format
                if (isinstance(call.get("function"), dict)
                    and call["function"].get("name")
                    and call["function"].get("arguments")):
                    result.fc_score += 10

        elif "name" in parsed and "arguments" in parsed:
            result.has_name = True
            result.has_arguments = True
            result.fc_score += 40
            result.issues.append("Flat format (no tool_calls wrapper)")

        elif "function_call" in parsed:
            result.has_tool_calls = True
            result.fc_score += 30
            result.issues.append("Legacy function_call format")

        elif "content" in parsed:
            result.fc_score += 5
            result.issues.append("JSON text-only response")

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract JSON from text."""
        if not text:
            return None
        text = text.strip()

        # Pure JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Code block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Bracket match
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == '\\' and in_str:
                esc = True
                continue
            if c == '"':
                in_str = not in_str
            elif not in_str:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except (json.JSONDecodeError, ValueError):
                            return None

        return None


# ═══════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════

class ReportGenerator:
    """Generate final ranking report."""

    def generate(
        self,
        scanner: G4FScanner,
        results: List[TestResult],
    ) -> List[CombinationScore]:
        """Generate ranked report."""

        print()
        print()
        print("=" * 70)
        print("  📊 FINAL RANKING REPORT")
        print("=" * 70)

        # Group results by provider+model
        combos: Dict[str, CombinationScore] = {}

        for r in results:
            key = f"{r.provider}||{r.model}"
            if key not in combos:
                combos[key] = CombinationScore(
                    provider=r.provider,
                    model=r.model,
                )

            combo = combos[key]

            if r.test_type == "fc":
                combo.fc_score = r.fc_score
                combo.fc_response_preview = r.response[:200]
                combo.avg_time = r.time_seconds

            elif r.test_type == "chat":
                combo.chat_ok = r.success
                combo.avg_time = (combo.avg_time + r.time_seconds) / 2

            elif r.test_type == "tool_result":
                combo.tool_ok = r.success

        # Calculate total score
        for combo in combos.values():
            combo.total_score = combo.fc_score
            if combo.chat_ok:
                combo.total_score += 30
            if combo.tool_ok:
                combo.total_score += 30

        # Sort
        ranked = sorted(
            combos.values(),
            key=lambda c: (-c.total_score, c.avg_time)
        )

        # ── FC Ranking ──────────────────────────────
        print("\n  🏆 TOP 15: FUNCTION CALLING")
        print("  " + "-" * 66)
        print(f"  {'#':3s} {'Provider':22s} {'Model':25s} {'FC':4s} {'Chat':5s} {'Tool':5s} {'Total':6s} {'Time':5s}")
        print("  " + "-" * 66)

        for i, c in enumerate(ranked[:15], 1):
            if c.fc_score == 0:
                continue
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}"
            chat = "✅" if c.chat_ok else "❌"
            tool = "✅" if c.tool_ok else "❌"
            print(
                f"  {emoji:3s} {c.provider:22s} {c.model:25s} "
                f"{c.fc_score:3d}  {chat:4s}  {tool:4s}  "
                f"{c.total_score:4d}   {c.avg_time:4.1f}s"
            )

        # ── Category breakdown ──────────────────────
        print("\n  📋 BEST PER MODEL CATEGORY:")
        print("  " + "-" * 66)

        categories_best = {}
        for c in ranked:
            if c.fc_score == 0:
                continue
            model_info = scanner.models.get(c.model, ModelInfo(c.model))
            cat = model_info.category
            if cat not in categories_best:
                categories_best[cat] = c

        for cat in ["flagship", "reasoning", "code", "fast", "open", "other"]:
            if cat in categories_best:
                c = categories_best[cat]
                print(
                    f"     {cat:12s} → {c.provider:20s} + {c.model:20s} "
                    f"(FC:{c.fc_score}, Total:{c.total_score})"
                )
            else:
                print(f"     {cat:12s} → No working combination")

        # ── Failed providers ────────────────────────
        print("\n  💀 FAILED PROVIDERS (0 score):")
        failed = set()
        for r in results:
            if not r.success:
                failed.add(f"{r.provider}: {r.error[:40]}")

        for f in sorted(failed)[:15]:
            print(f"     ✗ {f}")

        if len(failed) > 15:
            print(f"     ... and {len(failed)-15} more")

        # ── Recommendation ──────────────────────────
        print()
        print("=" * 70)
        print("  🎯 RECOMMENDATION FOR PICOCLAW")
        print("=" * 70)

        if ranked and ranked[0].fc_score > 0:
            best = ranked[0]
            print(f"""
  BEST COMBINATION:
    Provider: {best.provider}
    Model:    {best.model}
    FC Score: {best.fc_score}/100
    Chat:     {'✅' if best.chat_ok else '❌'}
    Tool:     {'✅' if best.tool_ok else '❌'}
    Speed:    {best.avg_time:.1f}s

  CONFIG (paste to config.json):
  {{
    "provider_lock": {{
      "strict_provider_mode": true,
      "locked_provider": "{best.provider}",
      "locked_model": "{best.model}",
      "fail_on_lock_error": false
    }}
  }}

  FC Response Sample:
  {best.fc_response_preview[:200]}""")

            if best.fc_score < 50:
                print("""
  ⚠️  WARNING: Best FC score is below 50/100
  Function Calling may not be reliable.
  Consider:
  1. Adding Groq free API as primary FC provider
     → https://console.groq.com/keys (free, email only)
  2. Using prompt engineering tricks in bridge""")

            # Show top 3 alternatives
            print("\n  ALTERNATIVES:")
            for i, c in enumerate(ranked[1:4], 2):
                if c.fc_score > 0:
                    print(f"    #{i}: {c.provider} + {c.model} (FC:{c.fc_score})")

        else:
            print("""
  ❌ NO PROVIDERS PASSED FC TEST!
  
  Options:
  1. Use Groq free API (native FC, 14k req/day)
     → https://console.groq.com/keys
  2. Update g4f: pip install --upgrade g4f
  3. Try again later (providers change daily)""")

        print()

        # ── Save ────────────────────────────────────
        self._save_results(scanner, ranked)

        return ranked

    def _save_results(
        self,
        scanner: G4FScanner,
        ranked: List[CombinationScore],
    ) -> None:
        """Save results to file."""
        output_dir = Path.home() / "g4f-bridge" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "full_scan_results.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "g4f_version": "",
            "total_models": len(scanner.models),
            "total_providers": len(scanner.providers),
            "ranking": [
                {
                    "provider": c.provider,
                    "model": c.model,
                    "fc_score": c.fc_score,
                    "chat_ok": c.chat_ok,
                    "tool_ok": c.tool_ok,
                    "total_score": c.total_score,
                    "avg_time": c.avg_time,
                }
                for c in ranked[:20]
            ],
            "recommendation": {
                "provider": ranked[0].provider if ranked else "",
                "model": ranked[0].model if ranked else "",
            } if ranked and ranked[0].fc_score > 0 else None,
            "compatibility": {
                model: providers
                for model, providers in list(scanner.compatibility.items())[:50]
            },
        }

        try:
            import g4f
            data["g4f_version"] = getattr(g4f, '__version__', 'unknown')
        except Exception:
            pass

        try:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  📁 Results saved: {output_file}")
        except Exception as e:
            print(f"  ⚠️  Save failed: {e}")

        print()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    # Parse args
    scan_only = "--scan-only" in sys.argv
    fast_mode = "--fast" in sys.argv
    smart_mode = "--smart" in sys.argv

    mode = "fast" if fast_mode else "smart" if smart_mode else "full"

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     g4f FULL SCAN & FUNCTION CALLING TEST                   ║")
    print(f"║     Mode: {mode:10s}                                        ║")
    print(f"║     Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):19s}                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Scan
    scanner = G4FScanner()
    num_models, num_providers = scanner.scan()
    scanner.print_summary()

    if scan_only:
        print("  (Scan-only mode, skipping tests)")
        return

    if num_providers == 0:
        print("  ERROR: No providers found!")
        return

    # Step 2: Select combinations
    tester = G4FTester()
    combinations = tester.select_combinations(scanner, mode)

    if not combinations:
        print("  ERROR: No valid combinations to test!")
        return

    # Estimate time
    est_time = len(combinations) * 3 * 15  # 3 tests × 15s avg
    print(f"  ⏱️  Estimated time: {est_time//60} - {est_time//30} minutes")
    print()

    try:
        input("  Press ENTER to start testing (Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        return

    # Step 3: Run tests
    results = tester.run_tests(
        scanner, combinations,
        skip_chat=(mode == "fast"),
        skip_tool=(mode == "fast"),
    )

    # Step 4: Generate report
    report = ReportGenerator()
    ranked = report.generate(scanner, results)

    print("=" * 70)
    print("  ✅ ALL DONE!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()