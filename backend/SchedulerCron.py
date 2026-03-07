"""
SchedulerCron.py — Standalone task scheduler (no GUI required)
==============================================================

This script is designed to be called by Windows Task Scheduler every hour.
It does ONE job: check if any pipeline tasks are overdue and, if so, write
them to the Firestore run_commands queue.  The Local Agent GUI then picks
them up and executes the heavy ML work.

This replaces Firebase Cloud Functions (which require the Blaze/paid plan).

Setup (run SetupScheduler.bat once, or see manual steps at the bottom):
    Double-click SetupScheduler.bat  — registers a Windows Task Scheduler
    task called "PortfolioManagerScheduler" that runs this script hourly.

What this covers:
  ✓ incremental_update  — weekly ML retrain
  ✓ connected_stocks    — biweekly connected-stock evaluation
  ✓ strategy_refresh    — biweekly strategy regeneration
  ✗ sentiment_fetch     — NOT queued here; needs Python ML libs, handled by
                          the Local Agent directly (startup catch-up + loop)

When your laptop is off:
  Nothing runs (unavoidable without a paid cloud service).
  When you turn the laptop back on:
    1. This script fires within the hour → queues any overdue ML tasks
    2. Local Agent startup catch-up → runs sentiment fetch immediately
    3. Open the agent → process the queued tasks
"""

from __future__ import annotations

import os
import sys
import datetime

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo('America/New_York')
except ImportError:
    _ET = None


def _now() -> datetime.datetime:
    return datetime.datetime.now(_ET) if _ET else datetime.datetime.now()

# ── Locate project root and load .env ────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

_env_path = os.path.join(PROJECT_DIR, '.env')
try:
    from dotenv import load_dotenv
    if os.path.isfile(_env_path):
        load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass

import firebase_admin
from firebase_admin import credentials, firestore

# ── Default intervals (hours) — must match LocalAgent.py defaults ─────────────
DEFAULT_SCHEDULE = {
    'incremental_update':    168,   # 7 days
    'connected_stocks':     336,   # 14 days
    'strategy_refresh':     336,   # 14 days
    'trading_execute_1h':   1,    # hourly
    'trading_execute_1d':   24,   # daily
    'daily_review':         24,   # daily performance review (fallback)
}

TASK_LABELS = {
    'incremental_update':    'Scheduled (cron): Incremental ML update',
    'connected_stocks':      'Scheduled (cron): Connected stocks evaluation',
    'strategy_refresh':      'Scheduled (cron): Strategy refresh',
    'trading_execute_1h':    'Scheduled (cron): Trading bot cycle [1h]',
    'trading_execute_1d':    'Scheduled (cron): Trading bot cycle [1d]',
    'daily_review':          'Scheduled (cron): Daily performance review',
}

_COL_COMMANDS  = 'run_commands'
_COL_STATE     = 'scheduler_state'
_COL_CONFIG    = 'scheduler_config'


def _init_firestore():
    sa_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH') or os.path.join(
        PROJECT_DIR,
        'protfoliomanagerv2-firebase-adminsdk-fbsvc-9fbdf566d1.json',
    )
    if not os.path.isfile(sa_path):
        raise FileNotFoundError(f"Firebase service account key not found: {sa_path}")
    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def _is_overdue(last_run_iso: str, interval_hours: float, now: datetime.datetime) -> bool:
    if not last_run_iso:
        return True
    try:
        last = datetime.datetime.fromisoformat(last_run_iso)
        if last.tzinfo is None and now.tzinfo is not None:
            last = last.replace(tzinfo=datetime.timezone.utc)
        elif last.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)
        elapsed = (now - last).total_seconds() / 3600
        return elapsed >= interval_hours
    except Exception:
        return True


def _hours_until_due(last_run_iso: str, interval_hours: float, now: datetime.datetime) -> float:
    if not last_run_iso:
        return 0.0
    try:
        last = datetime.datetime.fromisoformat(last_run_iso)
        if last.tzinfo is None and now.tzinfo is not None:
            last = last.replace(tzinfo=datetime.timezone.utc)
        elif last.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)
        elapsed = (now - last).total_seconds() / 3600
        return max(0.0, interval_hours - elapsed)
    except Exception:
        return 0.0


def _queue_task(db, task_type: str, description: str) -> bool:
    """Write a task to run_commands if one isn't already queued. Returns True if written."""
    existing = (
        db.collection(_COL_COMMANDS)
        .where('type',   '==', task_type)
        .where('status', '==', 'queued')
        .limit(1)
        .stream()
    )
    for _ in existing:
        print(f"  [{task_type}] already queued — skipping")
        return False

    # Get current portfolio stocks
    stock_docs = (
        db.collection('stock_metadata')
        .where('inPortfolio', '==', True)
        .stream()
    )
    stocks = [d.to_dict().get('symbol', d.id) for d in stock_docs if d.to_dict().get('symbol') or d.id]

    db.collection(_COL_COMMANDS).add({
        'type':        task_type,
        'status':      'queued',
        'description': description,
        'stocks':      stocks,
        'config':      {},
        'source':      'windows_task_scheduler',
        'createdAt':   _now().isoformat(),
        'approvedAt':  None,
        'startedAt':   None,
        'completedAt': None,
        'result':      None,
        'error':       None,
    })
    print(f"  [{task_type}] queued ✓ ({len(stocks)} stocks)")
    return True


def main():
    print(f"[SchedulerCron] {_now().isoformat()} ET")

    db  = _init_firestore()
    now = _now()

    # Load user-configured intervals from Firestore (set via Settings page)
    intervals = {**DEFAULT_SCHEDULE}
    try:
        config_doc = db.collection(_COL_CONFIG).document('intervals').get()
        if config_doc.exists:
            cfg = config_doc.to_dict()
            for key in DEFAULT_SCHEDULE:
                if isinstance(cfg.get(key), (int, float)) and cfg[key] > 0:
                    intervals[key] = cfg[key]
            print(f"[SchedulerCron] Loaded custom intervals: {intervals}")
    except Exception as e:
        print(f"[SchedulerCron] Could not load custom intervals: {e}")

    # Load last-run timestamps
    state = {}
    try:
        state_doc = db.collection(_COL_STATE).document('current').get()
        if state_doc.exists:
            state = state_doc.to_dict() or {}
    except Exception as e:
        print(f"[SchedulerCron] Could not load scheduler state: {e}")

    # Check app_config/trading scheduleMode — if 'automatic', skip queueing
    # trading tasks (LocalAgent runs them directly when it's running)
    trading_schedule_mode = 'queued'
    try:
        trading_doc = db.collection('app_config').document('trading').get()
        if trading_doc.exists:
            trading_schedule_mode = (trading_doc.to_dict() or {}).get('scheduleMode', 'queued')
    except Exception:
        pass

    # Check and queue overdue tasks
    state_updates = {}
    TRADING_TASKS = {'trading_execute_1h', 'trading_execute_1d'}
    for task_type, interval_hours in intervals.items():
        if task_type in TRADING_TASKS and trading_schedule_mode == 'automatic':
            print(f"  [{task_type}] SKIP — trading runs automatically (no queue)")
            continue
        last_key  = f'last_{task_type}'
        last_run  = state.get(last_key, '')
        if _is_overdue(last_run, interval_hours, now):
            print(f"  [{task_type}] OVERDUE — queueing...")
            queued = _queue_task(db, task_type, TASK_LABELS[task_type])
            if queued:
                state_updates[last_key] = now.isoformat()
        else:
            hours_left = _hours_until_due(last_run, interval_hours, now)
            print(f"  [{task_type}] OK — due in {hours_left:.1f}h")

    # Persist updated timestamps
    if state_updates:
        db.collection(_COL_STATE).document('current').set(state_updates, merge=True)
        print(f"[SchedulerCron] Updated state for: {list(state_updates.keys())}")

    print("[SchedulerCron] Done.")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        print(f"[SchedulerCron] FATAL: {exc}")
        traceback.print_exc()
        sys.exit(1)
