"""
Auto-updater for g4f with rollback capability.
"""

import os
import sys
import subprocess
import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from logger_setup import get_logger


class G4FUpdater:
    PYPI_URL = "https://pypi.org/pypi/g4f/json"

    def __init__(self) -> None:
        from environment import get_environment
        from config import get_config

        self.logger = get_logger("updater")
        self.env = get_environment()
        self.config = get_config()

        self.current_version: str = ""
        self.latest_version: str = ""
        self.last_check: Optional[datetime] = None
        self.is_updating: bool = False

    def get_current_version(self) -> str:
        try:
            import g4f  # type: ignore
            self.current_version = str(getattr(g4f, "__version__", "unknown"))
        except ImportError:
            self.current_version = "not_installed"
        return self.current_version

    async def check_latest_version(self) -> Tuple[bool, str]:
        loop = asyncio.get_event_loop()

        # Try aiohttp
        try:
            import aiohttp  # type: ignore
            async with aiohttp.ClientSession() as session:
                async with session.get(self.PYPI_URL, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        self.latest_version = data.get("info", {}).get("version", "")
                        self.last_check = datetime.now()
                        return True, self.latest_version
                    return False, f"HTTP {resp.status}"
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"aiohttp check failed: {e}")

        # Fallback: requests in executor
        try:
            import requests  # type: ignore

            def _check() -> Tuple[bool, str]:
                r = requests.get(self.PYPI_URL, timeout=15)
                if r.status_code == 200:
                    ver = r.json().get("info", {}).get("version", "")
                    return True, ver
                return False, f"HTTP {r.status_code}"

            ok, ver = await loop.run_in_executor(None, _check)
            if ok:
                self.latest_version = ver
                self.last_check = datetime.now()
            return ok, ver
        except Exception as e:
            return False, str(e)

    def _compare_versions(self, v1: str, v2: str) -> int:
        try:
            p1 = [int(x) for x in v1.split(".")[:4]]
            p2 = [int(x) for x in v2.split(".")[:4]]
            while len(p1) < 4:
                p1.append(0)
            while len(p2) < 4:
                p2.append(0)
            for a, b in zip(p1, p2):
                if a < b:
                    return -1
                if a > b:
                    return 1
            return 0
        except (ValueError, AttributeError):
            return -1 if v1 != v2 else 0

    def is_update_available(self) -> bool:
        if not self.current_version or not self.latest_version:
            return False
        if self.current_version in ("unknown", "not_installed"):
            return True
        return self._compare_versions(self.current_version, self.latest_version) < 0

    async def update(self, force: bool = False) -> Tuple[bool, str]:
        if self.is_updating:
            return False, "Update already in progress"

        self.is_updating = True
        old_version = self.get_current_version()

        try:
            ok, latest = await self.check_latest_version()
            if not ok:
                return False, f"Cannot check version: {latest}"

            if not force and not self.is_update_available():
                return True, f"Already at latest: {self.current_version}"

            self.logger.info(f"Updating g4f: {old_version} → {latest}")
            ok, msg = await self._pip_upgrade()

            if ok:
                new_ver = self._verify_update()
                self._save_record(old_version, new_ver or latest, "success")

                try:
                    from scanner import get_scanner
                    get_scanner().scan()
                except Exception:
                    pass

                return True, f"Updated to {new_ver or latest}"
            else:
                self.logger.error(f"Update failed: {msg}")
                if old_version not in ("unknown", "not_installed"):
                    await self._rollback(old_version)
                self._save_record(old_version, latest, "failed", msg)
                return False, msg
        finally:
            self.is_updating = False

    async def _pip_upgrade(self) -> Tuple[bool, str]:
        loop = asyncio.get_event_loop()

        def _run() -> Tuple[bool, str]:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "g4f"],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    return True, "OK"
                return False, (result.stderr or result.stdout)[:300]
            except subprocess.TimeoutExpired:
                return False, "Timeout (5min)"
            except Exception as e:
                # Termux fallback
                try:
                    rc = os.system("pip install --upgrade g4f")
                    return rc == 0, f"os.system returned {rc}"
                except Exception as e2:
                    return False, str(e2)

        return await loop.run_in_executor(None, _run)

    def _verify_update(self) -> Optional[str]:
        try:
            import importlib
            import g4f  # type: ignore
            importlib.reload(g4f)
            return str(getattr(g4f, "__version__", "unknown"))
        except Exception:
            return None

    async def _rollback(self, version: str) -> None:
        self.logger.warning(f"Rolling back to {version}")
        loop = asyncio.get_event_loop()

        def _run() -> None:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"g4f=={version}"],
                capture_output=True, timeout=300,
            )

        try:
            await loop.run_in_executor(None, _run)
            self._save_record(self.latest_version, version, "rolled_back")
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")

    def _save_record(self, from_v: str, to_v: str, status: str, error: Optional[str] = None) -> None:
        try:
            from storage import get_storage
            get_storage().add_update_record(from_v, to_v, status, error)
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_version": self.current_version or self.get_current_version(),
            "latest_version": self.latest_version,
            "update_available": self.is_update_available(),
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "is_updating": self.is_updating,
        }


class UpdateScheduler:
    def __init__(self) -> None:
        from config import get_config

        self.logger = get_logger("update_scheduler")
        self.config = get_config()
        self.updater = G4FUpdater()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if not self.config.updater.auto_update_enabled:
            self.logger.info("Auto-update disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        self.logger.info(f"Update scheduler started ({self.config.updater.check_interval_hours}h)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        interval = self.config.updater.check_interval_hours * 3600
        while self._running:
            try:
                ok, ver = await self.updater.check_latest_version()
                if ok and self.updater.is_update_available():
                    self.logger.info(f"Update available → {ver}")
                    ok, msg = await self.updater.update()
                    self.logger.info(f"Auto-update: {msg}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break


_updater: Optional[G4FUpdater] = None
_scheduler: Optional[UpdateScheduler] = None


def get_updater() -> G4FUpdater:
    global _updater
    if _updater is None:
        _updater = G4FUpdater()
    return _updater


def get_update_scheduler() -> UpdateScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = UpdateScheduler()
    return _scheduler