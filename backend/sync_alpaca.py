"""
Standalone Alpaca → Firestore Sync
==================================

Fetches account (equity, day P&L) and positions from Alpaca,
writes to Firestore broker_account/live and trade_positions.

Run without LocalAgent — useful when you only need Live Portfolio Value
in the UI without running the full agent.

Usage:
  python backend/sync_alpaca.py
  Or double-click SyncAlpaca.bat
"""

from __future__ import annotations

import os
import sys
import datetime

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(PROJECT_DIR, '.env')
    if os.path.isfile(_env_path):
        load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass

import firebase_admin
from firebase_admin import credentials, firestore
from BrokerClient import AlpacaBroker


def _init_firestore():
    sa_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
    if not sa_path:
        sa_path = os.path.join(
            PROJECT_DIR,
            'protfoliomanagerv2-firebase-adminsdk-fbsvc-9fbdf566d1.json',
        )
    if not os.path.isfile(sa_path):
        raise FileNotFoundError(f"Firebase service account not found: {sa_path}")
    if not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def _sync_account(broker, db, log):
    account = broker.get_account()
    db.collection('broker_account').document('live').set({
        'equity': account.equity,
        'cash': account.cash,
        'buyingPower': account.buyingPower,
        'portfolioValue': account.portfolioValue,
        'dayPL': account.dayPL,
        'dayPLPct': account.dayPLPct,
        'updatedAt': datetime.datetime.now().isoformat(),
        'brokerMode': broker.mode,
    }, merge=True)
    log(f"Synced account: equity=${account.equity:,.0f}, day P&L ${account.dayPL:+,.2f}")


def _sync_positions(broker, db, log):
    positions = broker.get_positions()
    col = db.collection('trade_positions')
    active_symbols = set()
    for p in positions:
        qty = int(p.qty) if p.qty else 0
        if qty <= 0:
            continue
        active_symbols.add(p.symbol)
        cost_basis = (p.avgEntryPrice or 0) * qty
        col.document(p.symbol).set({
            'symbol': p.symbol,
            'qty': qty,
            'avgEntryPrice': float(p.avgEntryPrice or 0),
            'currentPrice': float(p.currentPrice or 0),
            'marketValue': float(p.marketValue or 0),
            'unrealizedPL': float(p.unrealizedPL or 0),
            'unrealizedPLPct': float(p.unrealizedPLPct or 0),
            'totalInvested': cost_basis,
            'totalRealized': 0.0,
            'numBuys': 0,
            'numSells': 0,
            'lastUpdated': datetime.datetime.now().isoformat(),
        }, merge=True)
    for doc in col.stream():
        if doc.id not in active_symbols:
            col.document(doc.id).update({'qty': 0})
    log(f"Synced {len(active_symbols)} positions")


def main():
    def log(msg):
        print(msg)

    try:
        broker = AlpacaBroker()
    except ValueError as e:
        print(f"Error: {e}")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to connect to Alpaca: {e}")
        sys.exit(1)

    try:
        db = _init_firestore()
    except Exception as e:
        print(f"Failed to connect to Firestore: {e}")
        sys.exit(1)

    try:
        _sync_account(broker, db, log)
        _sync_positions(broker, db, log)
        print("Done. Refresh the Trading page to see Live Portfolio Value.")
    except Exception as e:
        print(f"Sync failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
