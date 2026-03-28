"""
Background Provider Test Worker.
Tests every g4f provider+model combination for:
  - Chat capability
  - Function Calling capability (with FC score)
  - Tool Result processing

Results are persisted to provider_tests.json and used by the router
to rank providers intelligently.
"""

import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from logger_setup import get_logger


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class TestEntry:
    """Test result for one provider+model combination."""
    provider: str = ""
    model: str = ""
    status: str = "untested"  # active|fc_capable|chat_only|unstable|dead|untested
    chat_ok: bool = False
    chat_time_ms: int = 0
    fc_ok: bool = False
    fc_time_ms: int = 0
    fc_score: int = 0
    tool_result_ok: bool = False
    success_streak: int = 0
    fail_streak: int = 0
    last_error: Optional[str] = None
    tested_at: Optional[str] = None
    next_test_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# FC Test Messages
# ═══════════════════════════════════════════════════════════════

_FC_SYSTEM = (
    "You are a function-calling AI. Respond ONLY with valid JSON.\n"
    'To call a function: {"tool_calls":[{"name":"get_weather","arguments":{"location":"Tokyo"}}]}\n'
    "NO text. NO markdown. ONLY JSON."
)

_FC_MESSAGES = [
    {"role": "system", "content": _FC_SYSTEM},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

_FC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

_CHAT_MESSAGES = [{"role": "user", "content": "Say OK"}]

_TOOL_RESULT_MESSAGES = [
    {"role": "user", "content": "What is the weather in Tokyo?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_test123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location":"Tokyo"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_test123",
        "content": '{"temperature": 22, "condition": "sunny"}',
    },
]

# For g4f compatibility (no role:"tool"), convert tool result messages
_TOOL_RESULT_MESSAGES_G4F = [
    {"role": "user", "content": "What is the weather in Tokyo?"},
    {"role": "assistant", "content": "I called get_weather for Tokyo."},
    {
        "role": "user",
        "content": (
            '[Tool Result for get_weather]:\n'
            '{"temperature": 22, "condition": "sunny"}\n\n'
            "Based on this result, tell me the weather in one sentence."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════
# Test Worker
# ═══════════════════════════════════════════════════════════════

class ProviderTestWorker:
    """Background worker that tests g4f provider+model combinations."""

    def __init__(self) -> None:
        from config import get_config
        from environment import get_environment

        self.logger = get_logger("test_worker")
        self.config = get_config()
        self.test_cfg = self.config.testing
        env = get_environment()

        self.results_file = env.data_dir / "provider_tests.json"
        self.results: Dict[str, TestEntry] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────

    def load_results(self) -> None:
        """Load previous results from JSON file."""
        if not self.results_file.exists():
            self.logger.info("No previous test results found")
            return

        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            tests = data.get("tests", {})
            for key, entry_data in tests.items():
                try:
                    # Filter valid fields only
                    valid = {
                        f.name
                        for f in TestEntry.__dataclass_fields__.values()
                    }
                    filtered = {
                        k: v for k, v in entry_data.items() if k in valid
                    }
                    self.results[key] = TestEntry(**filtered)
                except TypeError:
                    pass

            self.logger.info(
                f"Loaded {len(self.results)} test results from file"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load test results: {e}")

    def save_results(self) -> None:
        """Save results to JSON file."""
        with self._lock:
            summary = self._compute_summary()
            data = {
                "version": 2,
                "last_scan": datetime.utcnow().isoformat() + "Z",
                "g4f_version": self._get_g4f_version(),
                "tests": {
                    key: asdict(entry)
                    for key, entry in self.results.items()
                },
                "summary": summary,
            }

        try:
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save test results: {e}")

    async def start_async(self) -> None:
        """Start background worker (asyncio mode — for FastAPI)."""
        if not self.test_cfg.enabled or not self.test_cfg.background_worker:
            self.logger.info("Background testing disabled")
            return

        self.load_results()
        self._running = True
        self._task = asyncio.create_task(self._loop())
        self.logger.info("Background test worker started (asyncio)")

    def start_thread(self) -> None:
        """Start background worker (thread mode — for Flask)."""
        if not self.test_cfg.enabled or not self.test_cfg.background_worker:
            return

        self.load_results()
        self._running = True

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._loop())
            except Exception as e:
                self.logger.error(f"Test worker thread crashed: {e}")
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        self.logger.info("Background test worker started (thread)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def trigger_scan(self, force: bool = False) -> None:
        """Trigger a scan. If force=True, reset all next_test_at."""
        if force:
            with self._lock:
                for entry in self.results.values():
                    entry.next_test_at = None
            self.logger.info("Force scan: all timestamps reset")

    # ── Main Loop ─────────────────────────────────────────

    async def _loop(self) -> None:
        # Small delay to let server fully start
        await asyncio.sleep(5)
        self.logger.info("Test worker loop starting...")

        while self._running:
            try:
                await self._run_test_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Test cycle error: {e}")

            interval = self.test_cfg.scan_interval_minutes * 60
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    async def _run_test_cycle(self) -> None:
        from scanner import get_scanner

        scanner = get_scanner()
        if not scanner.providers:
            scanner.scan()

        if not scanner.providers:
            self.logger.warning("No g4f providers available for testing")
            return

        # Build combinations to test
        combinations = self._get_combinations_needing_test(scanner)

        if not combinations:
            self.logger.debug("No combinations need testing right now")
            return

        self.logger.info(
            f"Testing {len(combinations)} combinations..."
        )

        tested = 0
        for provider_name, model in combinations:
            if not self._running:
                break

            await self._test_combination(scanner, provider_name, model)
            tested += 1

            # Save periodically
            if tested % 10 == 0:
                self.save_results()

            await asyncio.sleep(self.test_cfg.sequential_delay_seconds)

        self.save_results()
        self.logger.info(f"Test cycle complete: {tested} tested")

    def _get_combinations_needing_test(
        self, scanner: Any
    ) -> List[Tuple[str, str]]:
        """Get provider+model pairs that need testing."""
        now = datetime.utcnow()
        intervals = self.test_cfg.retest_intervals_minutes
        need_test = []

        for pname, pinfo in scanner.providers.items():
            if pinfo.needs_auth:
                continue

            for model in pinfo.models:
                key = f"{pname}::{model}"
                entry = self.results.get(key)

                if entry is None:
                    # Never tested → test immediately
                    need_test.append((pname, model))
                    continue

                # Check retest interval
                status = entry.status
                interval_min = intervals.get(status, 360)

                if interval_min == 0:
                    # 0 means "test ASAP"
                    need_test.append((pname, model))
                    continue

                if entry.next_test_at:
                    try:
                        nxt = datetime.fromisoformat(
                            entry.next_test_at.replace("Z", "")
                        )
                        if now < nxt:
                            continue
                    except (ValueError, TypeError):
                        pass

                # Past due → add to list
                need_test.append((pname, model))

        # Sort: untested first, then unstable, then others
        priority = {
            "untested": 0, "unstable": 1, "chat_only": 2,
            "fc_capable": 3, "active": 4, "dead": 5,
        }
        need_test.sort(
            key=lambda x: priority.get(
                self.results.get(f"{x[0]}::{x[1]}", TestEntry()).status, 0
            )
        )

        return need_test

    # ── Single Combination Test ───────────────────────────

    async def _test_combination(
        self, scanner: Any, provider_name: str, model: str
    ) -> None:
        key = f"{provider_name}::{model}"
        entry = self.results.get(key, TestEntry(
            provider=provider_name, model=model
        ))

        self.logger.info(f"  Testing {key}...")

        # ── Chat Test ──
        chat_ok, chat_time = await self._test_chat(
            scanner, provider_name, model
        )
        entry.chat_ok = chat_ok
        entry.chat_time_ms = chat_time

        if not chat_ok:
            entry.fail_streak += 1
            entry.success_streak = 0

            if entry.fail_streak >= self.test_cfg.max_dead_streak:
                entry.status = "dead"
            else:
                entry.status = "unstable"

            self.logger.info(f"    Chat: ❌ ({entry.last_error})")
            self._update_timestamps(entry)
            with self._lock:
                self.results[key] = entry
            return

        self.logger.info(f"    Chat: ✅ ({chat_time}ms)")

        # ── FC Test ──
        fc_ok, fc_score, fc_time = await self._test_fc(
            scanner, provider_name, model
        )
        entry.fc_ok = fc_ok
        entry.fc_score = fc_score
        entry.fc_time_ms = fc_time

        if fc_ok:
            self.logger.info(f"    FC:   ✅ score={fc_score} ({fc_time}ms)")
        else:
            self.logger.info(f"    FC:   ❌ score={fc_score}")

        # ── Tool Result Test (only if FC passed) ──
        tool_ok = False
        if fc_ok:
            tool_ok, _ = await self._test_tool_result(
                scanner, provider_name, model
            )
            entry.tool_result_ok = tool_ok
            if tool_ok:
                self.logger.info("    Tool: ✅")
            else:
                self.logger.info("    Tool: ❌")

        # ── Determine Status ──
        entry.success_streak += 1
        entry.fail_streak = 0

        if chat_ok and fc_ok and tool_ok:
            entry.status = "active"
        elif chat_ok and fc_ok:
            entry.status = "fc_capable"
        elif chat_ok:
            entry.status = "chat_only"
        else:
            entry.status = "unstable"

        self._update_timestamps(entry)
        with self._lock:
            self.results[key] = entry

        self.logger.info(f"    → status={entry.status}")

    # ── Individual Tests ──────────────────────────────────

    async def _test_chat(
        self, scanner: Any, provider_name: str, model: str
    ) -> Tuple[bool, int]:
        """Returns (success, time_ms)."""
        try:
            t0 = time.time()
            response = await self._call_g4f(
                scanner, provider_name, model, _CHAT_MESSAGES
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            if response and response.strip():
                return True, elapsed_ms

            return False, elapsed_ms
        except Exception as e:
            # Store error in current entry context
            key = f"{provider_name}::{model}"
            entry = self.results.get(key, TestEntry(
                provider=provider_name, model=model
            ))
            entry.last_error = str(e)[:150]
            with self._lock:
                self.results[key] = entry
            return False, 0

    async def _test_fc(
        self, scanner: Any, provider_name: str, model: str
    ) -> Tuple[bool, int, int]:
        """Returns (success, fc_score, time_ms)."""
        try:
            t0 = time.time()
            response = await self._call_g4f(
                scanner, provider_name, model, _FC_MESSAGES
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            if not response or not response.strip():
                return False, 0, elapsed_ms

            score = self._calculate_fc_score(response)
            success = score >= 40  # Threshold
            return success, score, elapsed_ms

        except Exception:
            return False, 0, 0

    async def _test_tool_result(
        self, scanner: Any, provider_name: str, model: str
    ) -> Tuple[bool, int]:
        """Returns (success, time_ms)."""
        try:
            t0 = time.time()
            response = await self._call_g4f(
                scanner, provider_name, model, _TOOL_RESULT_MESSAGES_G4F
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            if response and response.strip() and len(response.strip()) > 5:
                return True, elapsed_ms

            return False, elapsed_ms
        except Exception:
            return False, 0

    # ── g4f Direct Call ───────────────────────────────────

    async def _call_g4f(
        self,
        scanner: Any,
        provider_name: str,
        model: str,
        messages: List[Dict[str, str]],
    ) -> str:
        """Call g4f provider directly (bypass router)."""
        pinfo = scanner.providers.get(provider_name)
        if not pinfo or not pinfo.class_ref:
            raise ValueError(f"Provider {provider_name} not found")

        import g4f  # type: ignore

        timeout = self.test_cfg.test_timeout_seconds

        # Try async first
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

        # Fallback to sync in executor
        loop = asyncio.get_event_loop()

        def _sync() -> str:
            return str(
                g4f.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    provider=pinfo.class_ref,
                )
            )

        return await asyncio.wait_for(
            loop.run_in_executor(None, _sync),
            timeout=timeout,
        )

    # ── FC Scoring ────────────────────────────────────────

    @staticmethod
    def _calculate_fc_score(text: str) -> int:
        """
        Score FC quality (0-100):
          30 pts: Pure JSON (no text wrapper)
          25 pts: Contains tool_calls array
          20 pts: Has valid name field
          15 pts: Has valid arguments
          10 pts: Nested function object (OpenAI format)
        """
        import re as _re

        score = 0
        text = text.strip()

        # Try parse JSON
        parsed = None

        # Pure JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                score += 30  # Pure JSON
        except (json.JSONDecodeError, ValueError):
            pass

        # JSON in code block
        if parsed is None:
            match = _re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        score += 10  # Wrapped JSON (partial credit)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Bracket matching fallback
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
                                    parsed = json.loads(text[start:i + 1])
                                    score += 10
                                except (json.JSONDecodeError, ValueError):
                                    pass
                                break

        if not parsed or not isinstance(parsed, dict):
            return score

        # Check tool_calls
        has_calls = False

        if "tool_calls" in parsed:
            score += 25
            calls = parsed["tool_calls"]
            if isinstance(calls, list) and calls:
                has_calls = True
                call = calls[0] if isinstance(calls[0], dict) else {}

                # Name check
                name = (
                    call.get("name")
                    or call.get("function_name")
                    or (
                        call.get("function", {}).get("name")
                        if isinstance(call.get("function"), dict) else None
                    )
                )
                if name:
                    score += 20

                # Arguments check
                args = (
                    call.get("arguments")
                    or call.get("parameters")
                    or call.get("params")
                    or (
                        call.get("function", {}).get("arguments")
                        if isinstance(call.get("function"), dict) else None
                    )
                )
                if args:
                    score += 15

                # OpenAI nested format bonus
                func = call.get("function")
                if (
                    isinstance(func, dict)
                    and func.get("name")
                    and func.get("arguments")
                ):
                    score += 10

        elif "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
            score += 40  # Flat format

        elif "function_call" in parsed:
            score += 30  # Legacy format

        return min(score, 100)

    # ── Helpers ───────────────────────────────────────────

    def _update_timestamps(self, entry: TestEntry) -> None:
        now = datetime.utcnow()
        entry.tested_at = now.isoformat() + "Z"

        intervals = self.test_cfg.retest_intervals_minutes
        interval_min = intervals.get(entry.status, 360)
        nxt = now + timedelta(minutes=interval_min)
        entry.next_test_at = nxt.isoformat() + "Z"

    def _compute_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {
            "total": 0, "active": 0, "fc_capable": 0,
            "chat_only": 0, "unstable": 0, "dead": 0, "untested": 0,
        }
        for entry in self.results.values():
            counts["total"] += 1
            s = entry.status
            if s in counts:
                counts[s] += 1
        return counts

    @staticmethod
    def _get_g4f_version() -> str:
        try:
            import g4f  # type: ignore
            return str(getattr(g4f, "__version__", "unknown"))
        except ImportError:
            return "not_installed"

    # ── Public Query API ──────────────────────────────────

    def get_ranked_providers(
        self, model: str, need_fc: bool = False
    ) -> List[Tuple[str, str, int]]:
        """
        Get providers ranked by quality for a given model.

        Args:
            model: Model name or "auto".
            need_fc: True if request has tools.

        Returns:
            List of (provider_name, model_name, fc_score) sorted best-first.
        """
        status_rank = {
            "active": 100, "fc_capable": 80,
            "chat_only": 50, "unstable": 20,
            "dead": 0, "untested": 10,
        }

        candidates = []

        with self._lock:
            for key, entry in self.results.items():
                # Filter by status
                if entry.status in ("dead",):
                    continue

                # Filter by model
                if model != "auto" and entry.model != model:
                    # Fuzzy match
                    if model.lower() not in entry.model.lower():
                        continue

                # If FC needed, skip chat_only
                if need_fc and entry.status == "chat_only":
                    continue

                # If FC needed and score too low, skip
                if need_fc and entry.fc_score < self.config.routing.prefer_fc_score_above:
                    if entry.status not in ("active", "fc_capable"):
                        continue

                rank = status_rank.get(entry.status, 0)
                if need_fc:
                    rank += entry.fc_score
                # Bonus for speed
                if entry.chat_time_ms > 0 and entry.chat_time_ms < 2000:
                    rank += 10

                candidates.append((
                    entry.provider, entry.model,
                    entry.fc_score, rank,
                ))

        # Sort by rank descending
        candidates.sort(key=lambda x: -x[3])
        return [(c[0], c[1], c[2]) for c in candidates]

    def get_results_dict(self) -> Dict[str, Any]:
        """Get results for API response."""
        with self._lock:
            return {
                "version": 2,
                "last_scan": datetime.utcnow().isoformat() + "Z",
                "total": len(self.results),
                "summary": self._compute_summary(),
                "tests": {
                    key: asdict(entry)
                    for key, entry in self.results.items()
                },
            }


# ═══════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════

_worker_lock = threading.Lock()
_worker: Optional[ProviderTestWorker] = None


def get_test_worker() -> ProviderTestWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = ProviderTestWorker()
    return _worker