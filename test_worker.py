"""
Provider Test Worker v2.
- Tests on startup (blocking)
- Retests every 24 hours (background)
- Auto-save every 10 tests
"""

import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from logger_setup import get_logger


@dataclass
class TestResult:
    provider: str = ""
    model: str = ""
    status: str = "untested"
    chat_ok: bool = False
    chat_time_ms: int = 0
    fc_ok: bool = False
    fc_time_ms: int = 0
    fc_score: int = 0
    tool_result_ok: bool = False
    last_error: Optional[str] = None
    tested_at: Optional[str] = None


_CHAT_MSG = [{"role": "user", "content": "Reply with exactly: OK"}]

_FC_SYSTEM = """You are a JSON-only assistant. Output ONLY valid JSON.
Format: {"tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"Tokyo"}}}]}
NO text. NO markdown. ONLY JSON."""

_FC_MSG = [
    {"role": "system", "content": _FC_SYSTEM},
    {"role": "user", "content": "Get weather in Paris"},
]

_TOOL_RESULT_MSG = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "I'll check the weather."},
    {
        "role": "user",
        "content": "[Tool Result]: Temperature is 18°C, cloudy.\n\nNow answer briefly.",
    },
]


class ProviderTestWorker:
    def __init__(self) -> None:
        from config import get_config
        from environment import get_environment

        self.logger = get_logger("test_worker")
        self.config = get_config()
        self.test_cfg = self.config.testing
        env = get_environment()

        self.results_file = env.data_dir / "provider_tests.json"
        self.results: Dict[str, TestResult] = {}
        self.working_combinations: Set[str] = set()
        self.tested_at: Optional[datetime] = None

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    def load_results(self) -> bool:
        if not self.results_file.exists():
            self.logger.info("No cached test results")
            return False

        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            tested_at_str = data.get("tested_at")
            if tested_at_str:
                try:
                    self.tested_at = datetime.fromisoformat(
                        tested_at_str.replace("Z", "")
                    )
                except (ValueError, TypeError):
                    pass

            if self.tested_at:
                age = datetime.utcnow() - self.tested_at
                max_age = timedelta(hours=self.test_cfg.retest_interval_hours)

                if age < max_age:
                    tests = data.get("tests", {})
                    for key, entry in tests.items():
                        valid = {
                            f.name
                            for f in TestResult.__dataclass_fields__.values()
                        }
                        filtered = {
                            k: v for k, v in entry.items() if k in valid
                        }
                        self.results[key] = TestResult(**filtered)

                    self._update_working_set()
                    self.logger.info(
                        f"✓ Loaded {len(self.working_combinations)} working "
                        f"(cache age: {age.total_seconds() / 3600:.1f}h)"
                    )
                    return True

            self.logger.info("Cache expired, will retest")
            return False

        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return False

    def save_results(self) -> None:
        with self._lock:
            working = [
                k for k, r in self.results.items()
                if r.status in ("active", "fc_capable", "chat_only")
            ]

            data = {
                "version": 3,
                "tested_at": datetime.utcnow().isoformat() + "Z",
                "g4f_version": self._get_g4f_version(),
                "working_count": len(working),
                "total_count": len(self.results),
                "working": working,
                "tests": {k: asdict(v) for k, v in self.results.items()},
            }

        try:
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save: {e}")

    def _update_working_set(self) -> None:
        with self._lock:
            self.working_combinations.clear()
            for key, r in self.results.items():
                if r.status in ("active", "fc_capable", "chat_only"):
                    self.working_combinations.add(key)

    async def run_startup_test(self) -> int:
        if not self.test_cfg.enabled:
            self.logger.info("Testing disabled")
            return 0

        if self.load_results() and self.working_combinations:
            return len(self.working_combinations)

        self.logger.info("=" * 50)
        self.logger.info("  STARTING PROVIDER TESTS")
        self.logger.info("=" * 50)

        await self._run_full_test()
        self._update_working_set()
        self.save_results()

        count = len(self.working_combinations)
        if count == 0:
            self.logger.error("⚠ NO WORKING PROVIDERS FOUND!")
        else:
            self.logger.info(f"✓ {count} working combinations")

        return count

    async def _run_full_test(self) -> None:
        from scanner import get_scanner

        scanner = get_scanner()
        if not scanner.providers:
            scanner.scan()

        if not scanner.providers:
            self.logger.error("No g4f providers!")
            return

        combinations = []
        for pname, pinfo in scanner.providers.items():
            if pinfo.needs_auth:
                continue
            for model in pinfo.models:
                combinations.append((pname, model, pinfo))

        total = len(combinations)
        self.logger.info(f"Testing {total} combinations...")

        tested = 0
        for pname, model, pinfo in combinations:
            tested += 1
            key = f"{pname}::{model}"

            self.logger.info(f"  [{tested}/{total}] {pname} :: {model}")

            result = await self._test_single(pinfo, model)
            result.provider = pname
            result.model = model
            result.tested_at = datetime.utcnow().isoformat() + "Z"

            with self._lock:
                self.results[key] = result

            icon = {
                "active": "✅", "fc_capable": "🔧",
                "chat_only": "💬", "dead": "❌",
            }.get(result.status, "❓")

            self.logger.info(f"      {icon} {result.status} (fc={result.fc_score})")

            if tested % 10 == 0:
                self._update_working_set()
                self.save_results()
                self.logger.info(f"  💾 Auto-saved ({tested}/{total})")

            await asyncio.sleep(self.test_cfg.sequential_delay_seconds)

        self.tested_at = datetime.utcnow()

    async def _test_single(self, pinfo: Any, model: str) -> TestResult:
        result = TestResult()
        timeout = self.test_cfg.test_timeout_seconds

        try:
            t0 = time.time()
            response = await self._call_g4f(pinfo, model, _CHAT_MSG, timeout)
            elapsed = int((time.time() - t0) * 1000)

            if response and response.strip():
                result.chat_ok = True
                result.chat_time_ms = elapsed
            else:
                result.last_error = "Empty response"
                result.status = "dead"
                return result

        except Exception as e:
            result.last_error = str(e)[:100]
            result.status = "dead"
            return result

        try:
            t0 = time.time()
            response = await self._call_g4f(pinfo, model, _FC_MSG, timeout)
            elapsed = int((time.time() - t0) * 1000)

            if response:
                result.fc_time_ms = elapsed
                result.fc_score = self._calc_fc_score(response)
                result.fc_ok = result.fc_score >= self.test_cfg.min_fc_score

        except Exception:
            result.fc_ok = False

        if result.fc_ok:
            try:
                response = await self._call_g4f(
                    pinfo, model, _TOOL_RESULT_MSG, timeout
                )
                if response and len(response.strip()) > 5:
                    result.tool_result_ok = True
            except Exception:
                pass

        if result.chat_ok and result.fc_ok and result.tool_result_ok:
            result.status = "active"
        elif result.chat_ok and result.fc_ok:
            result.status = "fc_capable"
        elif result.chat_ok:
            result.status = "chat_only"
        else:
            result.status = "dead"

        return result

    async def _call_g4f(
        self, pinfo: Any, model: str,
        messages: List[Dict], timeout: int,
    ) -> str:
        import g4f

        try:
            result = await asyncio.wait_for(
                g4f.ChatCompletion.create_async(
                    model=model,
                    messages=messages,
                    provider=pinfo.class_ref,
                ),
                timeout=timeout,
            )
            return str(result)
        except (AttributeError, TypeError):
            pass

        loop = asyncio.get_event_loop()

        def _sync() -> str:
            return str(g4f.ChatCompletion.create(
                model=model,
                messages=messages,
                provider=pinfo.class_ref,
            ))

        return await asyncio.wait_for(
            loop.run_in_executor(None, _sync),
            timeout=timeout,
        )

    @staticmethod
    def _calc_fc_score(text: str) -> int:
        import re

        score = 0
        text = text.strip()
        parsed = None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                score += 30
        except (json.JSONDecodeError, ValueError):
            pass

        if parsed is None:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        score += 15
                except (json.JSONDecodeError, ValueError):
                    pass

        if parsed is None:
            start = text.find('{')
            if start != -1:
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
                                    parsed = json.loads(text[start:i+1])
                                    score += 10
                                except (json.JSONDecodeError, ValueError):
                                    pass
                                break

        if not parsed or not isinstance(parsed, dict):
            return score

        if "tool_calls" in parsed:
            score += 25
            calls = parsed["tool_calls"]
            if isinstance(calls, list) and calls:
                call = calls[0] if isinstance(calls[0], dict) else {}

                name = (
                    call.get("name") or
                    call.get("function_name") or
                    call.get("tool") or
                    (call.get("function", {}).get("name")
                     if isinstance(call.get("function"), dict) else None)
                )
                if name:
                    score += 20

                args = (
                    call.get("arguments") or
                    call.get("parameters") or
                    (call.get("function", {}).get("arguments")
                     if isinstance(call.get("function"), dict) else None)
                )
                if args:
                    score += 15

                if isinstance(call.get("function"), dict):
                    if call["function"].get("name") and call["function"].get("arguments"):
                        score += 10

        elif "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
            score += 50

        return min(score, 100)

    async def start_scheduler(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        self.logger.info(f"Scheduler started ({self.test_cfg.retest_interval_hours}h)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self) -> None:
        interval = self.test_cfg.retest_interval_hours * 3600

        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break

                self.logger.info("🔄 24h retest triggered")
                await self._run_full_test()
                self._update_working_set()
                self.save_results()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")

    def is_working(self, provider: str, model: str) -> bool:
        key = f"{provider}::{model}"
        return key in self.working_combinations

    def get_working_providers(
        self, model: str, need_fc: bool = False
    ) -> List[Tuple[str, str, int]]:
        candidates = []

        with self._lock:
            for key in self.working_combinations:
                result = self.results.get(key)
                if not result:
                    continue

                if model != "auto":
                    if result.model != model:
                        if model.lower() not in result.model.lower():
                            continue

                if need_fc:
                    if result.status == "chat_only":
                        continue
                    if result.fc_score < self.config.testing.min_fc_score:
                        continue

                rank = result.fc_score
                if result.status == "active":
                    rank += 50
                elif result.status == "fc_capable":
                    rank += 30
                if result.chat_time_ms > 0 and result.chat_time_ms < 3000:
                    rank += 20

                candidates.append((
                    result.provider, result.model,
                    result.fc_score, rank,
                ))

        candidates.sort(key=lambda x: -x[3])
        return [(c[0], c[1], c[2]) for c in candidates]

    def get_working_models(self) -> List[str]:
        models = set()
        with self._lock:
            for key in self.working_combinations:
                result = self.results.get(key)
                if result:
                    models.add(result.model)
        return sorted(models)

    def get_results_dict(self) -> Dict[str, Any]:
        with self._lock:
            working = list(self.working_combinations)
            return {
                "version": 3,
                "tested_at": self.tested_at.isoformat() + "Z" if self.tested_at else None,
                "working_count": len(working),
                "total_count": len(self.results),
                "working": working,
                "tests": {k: asdict(v) for k, v in self.results.items()},
            }

    def get_summary(self) -> Dict[str, int]:
        counts = {"active": 0, "fc_capable": 0, "chat_only": 0, "dead": 0}
        with self._lock:
            for r in self.results.values():
                if r.status in counts:
                    counts[r.status] += 1
        counts["working"] = counts["active"] + counts["fc_capable"] + counts["chat_only"]
        counts["total"] = len(self.results)
        return counts

    @staticmethod
    def _get_g4f_version() -> str:
        try:
            import g4f
            return str(getattr(g4f, "__version__", "unknown"))
        except ImportError:
            return "not_installed"


_worker_lock = threading.Lock()
_worker: Optional[ProviderTestWorker] = None


def get_test_worker() -> ProviderTestWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = ProviderTestWorker()
    return _worker