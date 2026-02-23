#!/usr/bin/env python3
"""
BlackRoad Crypto Payment Processor
Production-grade crypto payment handling with BTC/ETH/SOL/USDC support,
mock confirmation engine, wallet summaries, and fraud detection.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import os
import random
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict

DB_PATH = os.path.expanduser("~/.blackroad/crypto_payments.db")

SUPPORTED_COINS = {"BTC", "ETH", "SOL", "USDC", "MATIC", "AVAX"}

# Mock USD prices (would come from a live feed in production)
COIN_USD_RATES: Dict[str, float] = {
    "BTC": 65000.0,
    "ETH": 3200.0,
    "SOL": 150.0,
    "USDC": 1.0,
    "MATIC": 0.85,
    "AVAX": 38.0,
}

# Seconds to simulate confirmation times per coin
CONFIRMATION_TIMES: Dict[str, int] = {
    "BTC": 600,   # ~10 min per block
    "ETH": 15,    # ~15 sec
    "SOL": 2,     # ~2 sec
    "USDC": 15,   # ERC-20
    "MATIC": 3,
    "AVAX": 2,
}

REQUIRED_CONFIRMATIONS: Dict[str, int] = {
    "BTC": 3,
    "ETH": 12,
    "SOL": 32,
    "USDC": 12,
    "MATIC": 64,
    "AVAX": 10,
}


@dataclass
class Payment:
    id: str
    from_wallet: str
    to_wallet: str
    amount: float
    coin: str
    tx_hash: str
    status: str           # pending | confirmed | failed | refunded
    confirmations: int
    required_confirmations: int
    fee_amount: float
    fee_coin: str
    created_at: str
    confirmed_at: Optional[str] = None
    usd_value: float = 0.0
    memo: str = ""
    network: str = "mainnet"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Wallet:
    address: str
    label: str
    coin: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.utcnow().isoformat()


def _mock_tx_hash(coin: str) -> str:
    if coin == "BTC":
        return "".join(random.choices("0123456789abcdef", k=64))
    elif coin in ("ETH", "USDC", "MATIC"):
        return "0x" + "".join(random.choices("0123456789abcdef", k=64))
    else:
        return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", k=88))


def _coin_to_usd(amount: float, coin: str) -> float:
    rate = COIN_USD_RATES.get(coin, 0.0)
    return round(amount * rate, 2)


def _estimate_fee(amount: float, coin: str) -> float:
    """Estimate a network fee based on coin type."""
    if coin == "BTC":
        return round(0.0001, 8)       # ~0.0001 BTC flat
    elif coin == "ETH":
        return round(amount * 0.002, 8)
    elif coin == "SOL":
        return 0.000005
    elif coin == "USDC":
        return round(2.5 / COIN_USD_RATES.get("ETH", 3200), 8)  # gas in ETH
    elif coin == "MATIC":
        return 0.01
    elif coin == "AVAX":
        return 0.001
    return round(amount * 0.001, 8)


def _mock_confirmations(created_at_str: str, coin: str) -> int:
    """Mock confirmation count based on elapsed time since creation."""
    created = datetime.fromisoformat(created_at_str)
    elapsed_seconds = (datetime.utcnow() - created).total_seconds()
    conf_time = CONFIRMATION_TIMES.get(coin, 60)
    if conf_time == 0:
        return 0
    confirmations = int(elapsed_seconds / conf_time)
    return min(confirmations, REQUIRED_CONFIRMATIONS.get(coin, 12) + 5)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(path: str = DB_PATH) -> None:
    with get_db(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS payments (
                id                    TEXT PRIMARY KEY,
                from_wallet           TEXT NOT NULL,
                to_wallet             TEXT NOT NULL,
                amount                REAL NOT NULL,
                coin                  TEXT NOT NULL,
                tx_hash               TEXT UNIQUE NOT NULL,
                status                TEXT NOT NULL DEFAULT 'pending',
                confirmations         INTEGER NOT NULL DEFAULT 0,
                required_confirmations INTEGER NOT NULL DEFAULT 12,
                fee_amount            REAL NOT NULL DEFAULT 0,
                fee_coin              TEXT NOT NULL,
                created_at            TEXT NOT NULL,
                confirmed_at          TEXT,
                usd_value             REAL NOT NULL DEFAULT 0,
                memo                  TEXT NOT NULL DEFAULT '',
                network               TEXT NOT NULL DEFAULT 'mainnet'
            );
            CREATE TABLE IF NOT EXISTS wallets (
                address    TEXT PRIMARY KEY,
                label      TEXT NOT NULL DEFAULT '',
                coin       TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pay_from    ON payments(from_wallet);
            CREATE INDEX IF NOT EXISTS idx_pay_to      ON payments(to_wallet);
            CREATE INDEX IF NOT EXISTS idx_pay_status  ON payments(status);
            CREATE INDEX IF NOT EXISTS idx_pay_coin    ON payments(coin);
        """)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def create_payment(
    from_wallet: str,
    to_wallet: str,
    amount: float,
    coin: str,
    memo: str = "",
    network: str = "mainnet",
    path: str = DB_PATH,
) -> Payment:
    """Create a new crypto payment in pending status."""
    coin = coin.upper()
    if coin not in SUPPORTED_COINS:
        raise ValueError(f"Unsupported coin: {coin}. Supported: {SUPPORTED_COINS}")
    if amount <= 0:
        raise ValueError("Payment amount must be positive")
    if from_wallet == to_wallet:
        raise ValueError("Cannot send to the same wallet")

    fee = _estimate_fee(amount, coin)
    usd_val = _coin_to_usd(amount, coin)
    req_conf = REQUIRED_CONFIRMATIONS.get(coin, 12)

    payment = Payment(
        id=str(uuid.uuid4()),
        from_wallet=from_wallet,
        to_wallet=to_wallet,
        amount=amount,
        coin=coin,
        tx_hash=_mock_tx_hash(coin),
        status="pending",
        confirmations=0,
        required_confirmations=req_conf,
        fee_amount=fee,
        fee_coin=coin,
        created_at=_now(),
        usd_value=usd_val,
        memo=memo,
        network=network,
    )
    with get_db(path) as conn:
        conn.execute(
            """INSERT INTO payments
               (id, from_wallet, to_wallet, amount, coin, tx_hash, status, confirmations,
                required_confirmations, fee_amount, fee_coin, created_at, usd_value, memo, network)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (payment.id, payment.from_wallet, payment.to_wallet, payment.amount, payment.coin,
             payment.tx_hash, payment.status, payment.confirmations, payment.required_confirmations,
             payment.fee_amount, payment.fee_coin, payment.created_at, payment.usd_value,
             payment.memo, payment.network),
        )
    return payment


def check_status(payment_id: str, path: str = DB_PATH) -> Payment:
    """Check and update payment status based on mock confirmation time."""
    with get_db(path) as conn:
        row = conn.execute("SELECT * FROM payments WHERE id=?", (payment_id,)).fetchone()
        if not row:
            raise KeyError(f"Payment {payment_id} not found")
        payment = _row_to_payment(row)
        if payment.status == "pending":
            new_confs = _mock_confirmations(payment.created_at, payment.coin)
            new_status = payment.status
            confirmed_at = payment.confirmed_at
            if new_confs >= payment.required_confirmations:
                new_status = "confirmed"
                confirmed_at = _now()
            conn.execute(
                "UPDATE payments SET confirmations=?, status=?, confirmed_at=? WHERE id=?",
                (new_confs, new_status, confirmed_at, payment_id),
            )
            payment.confirmations = new_confs
            payment.status = new_status
            payment.confirmed_at = confirmed_at
    return payment


def get_payment(payment_id: str, path: str = DB_PATH) -> Payment:
    with get_db(path) as conn:
        row = conn.execute("SELECT * FROM payments WHERE id=?", (payment_id,)).fetchone()
    if not row:
        raise KeyError(f"Payment {payment_id} not found")
    return _row_to_payment(row)


def get_wallet_summary(
    wallet_address: str,
    path: str = DB_PATH,
) -> Dict:
    """Get a summary of all transactions for a wallet address."""
    with get_db(path) as conn:
        sent_rows = conn.execute(
            "SELECT coin, SUM(amount) as total, COUNT(*) as cnt FROM payments WHERE from_wallet=? AND status='confirmed' GROUP BY coin",
            (wallet_address,),
        ).fetchall()
        received_rows = conn.execute(
            "SELECT coin, SUM(amount) as total, COUNT(*) as cnt FROM payments WHERE to_wallet=? AND status='confirmed' GROUP BY coin",
            (wallet_address,),
        ).fetchall()
        pending_rows = conn.execute(
            "SELECT COUNT(*) as cnt FROM payments WHERE (from_wallet=? OR to_wallet=?) AND status='pending'",
            (wallet_address, wallet_address),
        ).fetchone()

    sent = {r["coin"]: {"total": r["total"], "count": r["cnt"]} for r in sent_rows}
    received = {r["coin"]: {"total": r["total"], "count": r["cnt"]} for r in received_rows}

    # Calculate net balance per coin
    all_coins = set(list(sent.keys()) + list(received.keys()))
    balances = {}
    for coin in all_coins:
        recv = received.get(coin, {}).get("total", 0.0)
        snt = sent.get(coin, {}).get("total", 0.0)
        balances[coin] = {
            "net": round(recv - snt, 8),
            "received": round(recv, 8),
            "sent": round(snt, 8),
            "usd_net": _coin_to_usd(recv - snt, coin),
        }

    return {
        "wallet": wallet_address,
        "balances": balances,
        "pending_transactions": pending_rows["cnt"] if pending_rows else 0,
        "total_usd_value": round(sum(b["usd_net"] for b in balances.values() if b["usd_net"] > 0), 2),
    }


def detect_suspicious(
    threshold_usd: float = 10000.0,
    path: str = DB_PATH,
) -> List[Payment]:
    """Detect suspicious payments above threshold USD value."""
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM payments WHERE usd_value >= ? ORDER BY usd_value DESC",
            (threshold_usd,),
        ).fetchall()
    return [_row_to_payment(r) for r in rows]


def detect_rapid_payments(
    wallet: str,
    window_minutes: int = 60,
    min_count: int = 5,
    path: str = DB_PATH,
) -> List[Payment]:
    """Detect wallets making many payments in a short window (possible automation/fraud)."""
    cutoff = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat()
    with get_db(path) as conn:
        rows = conn.execute(
            """SELECT * FROM payments
               WHERE from_wallet=? AND created_at >= ?
               ORDER BY created_at DESC""",
            (wallet, cutoff),
        ).fetchall()
    payments = [_row_to_payment(r) for r in rows]
    if len(payments) >= min_count:
        return payments
    return []


def export_transactions(
    wallet: Optional[str] = None,
    fmt: str = "json",
    path: str = DB_PATH,
) -> str:
    """Export transactions as JSON or CSV."""
    with get_db(path) as conn:
        if wallet:
            rows = conn.execute(
                "SELECT * FROM payments WHERE from_wallet=? OR to_wallet=? ORDER BY created_at DESC",
                (wallet, wallet),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM payments ORDER BY created_at DESC").fetchall()
    payments = [_row_to_payment(r) for r in rows]

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "from_wallet", "to_wallet", "amount", "coin", "tx_hash",
            "status", "confirmations", "fee_amount", "usd_value", "created_at", "confirmed_at",
        ])
        for p in payments:
            writer.writerow([
                p.id, p.from_wallet, p.to_wallet, p.amount, p.coin, p.tx_hash,
                p.status, p.confirmations, p.fee_amount, p.usd_value, p.created_at, p.confirmed_at,
            ])
        return buf.getvalue()
    else:
        return json.dumps([p.to_dict() for p in payments], indent=2)


def list_payments(
    wallet: Optional[str] = None,
    coin: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    path: str = DB_PATH,
) -> List[Payment]:
    with get_db(path) as conn:
        query = "SELECT * FROM payments WHERE 1=1"
        params = []
        if wallet:
            query += " AND (from_wallet=? OR to_wallet=?)"
            params.extend([wallet, wallet])
        if coin:
            query += " AND coin=?"
            params.append(coin.upper())
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
    return [_row_to_payment(r) for r in rows]


def register_wallet(
    address: str,
    coin: str,
    label: str = "",
    path: str = DB_PATH,
) -> Wallet:
    coin = coin.upper()
    if coin not in SUPPORTED_COINS:
        raise ValueError(f"Unsupported coin: {coin}")
    wallet = Wallet(address=address, label=label, coin=coin, created_at=_now())
    with get_db(path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO wallets (address, label, coin, created_at) VALUES (?,?,?,?)",
            (wallet.address, wallet.label, wallet.coin, wallet.created_at),
        )
    return wallet


def network_stats(path: str = DB_PATH) -> Dict:
    """Return aggregate network statistics."""
    with get_db(path) as conn:
        total = conn.execute("SELECT COUNT(*) as cnt FROM payments").fetchone()["cnt"]
        by_coin = conn.execute(
            "SELECT coin, COUNT(*) as cnt, SUM(amount) as vol, SUM(usd_value) as usd_vol FROM payments GROUP BY coin"
        ).fetchall()
        by_status = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM payments GROUP BY status"
        ).fetchall()
        total_fees_usd = conn.execute(
            "SELECT SUM(fee_amount * ?) as total FROM payments WHERE coin='ETH'",
            (COIN_USD_RATES["ETH"],),
        ).fetchone()["total"] or 0

    return {
        "total_payments": total,
        "by_coin": {
            r["coin"]: {
                "count": r["cnt"],
                "volume": round(r["vol"], 8),
                "usd_volume": round(r["usd_vol"], 2),
            }
            for r in by_coin
        },
        "by_status": {r["status"]: r["cnt"] for r in by_status},
        "total_fees_usd_eth": round(total_fees_usd, 2),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _row_to_payment(row: sqlite3.Row) -> Payment:
    return Payment(
        id=row["id"], from_wallet=row["from_wallet"], to_wallet=row["to_wallet"],
        amount=row["amount"], coin=row["coin"], tx_hash=row["tx_hash"],
        status=row["status"], confirmations=row["confirmations"],
        required_confirmations=row["required_confirmations"],
        fee_amount=row["fee_amount"], fee_coin=row["fee_coin"],
        created_at=row["created_at"], confirmed_at=row["confirmed_at"],
        usd_value=row["usd_value"], memo=row["memo"], network=row["network"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(obj) -> None:
    if hasattr(obj, "to_dict"):
        print(json.dumps(obj.to_dict(), indent=2))
    elif isinstance(obj, list):
        print(json.dumps([o.to_dict() if hasattr(o, "to_dict") else o for o in obj], indent=2))
    else:
        print(json.dumps(obj, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crypto_payments", description="BlackRoad Crypto Payment Processor")
    parser.add_argument("--db", default=DB_PATH)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init")

    p = sub.add_parser("create")
    p.add_argument("from_wallet")
    p.add_argument("to_wallet")
    p.add_argument("amount", type=float)
    p.add_argument("coin")
    p.add_argument("--memo", default="")
    p.add_argument("--network", default="mainnet")

    p = sub.add_parser("status")
    p.add_argument("payment_id")

    p = sub.add_parser("get")
    p.add_argument("payment_id")

    p = sub.add_parser("wallet-summary")
    p.add_argument("wallet_address")

    p = sub.add_parser("suspicious")
    p.add_argument("--threshold", type=float, default=10000.0)

    p = sub.add_parser("rapid")
    p.add_argument("wallet")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--min-count", type=int, default=5)

    p = sub.add_parser("export")
    p.add_argument("--wallet", default=None)
    p.add_argument("--format", default="json", choices=["json", "csv"])

    p = sub.add_parser("list")
    p.add_argument("--wallet", default=None)
    p.add_argument("--coin", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("register-wallet")
    p.add_argument("address")
    p.add_argument("coin")
    p.add_argument("--label", default="")

    sub.add_parser("network-stats")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    db = args.db
    init_db(db)

    if args.command == "init":
        print("Database initialized.")
    elif args.command == "create":
        p = create_payment(args.from_wallet, args.to_wallet, args.amount, args.coin, args.memo, args.network, db)
        _print_json(p)
    elif args.command == "status":
        _print_json(check_status(args.payment_id, db))
    elif args.command == "get":
        _print_json(get_payment(args.payment_id, db))
    elif args.command == "wallet-summary":
        print(json.dumps(get_wallet_summary(args.wallet_address, db), indent=2))
    elif args.command == "suspicious":
        _print_json(detect_suspicious(args.threshold, db))
    elif args.command == "rapid":
        _print_json(detect_rapid_payments(args.wallet, args.window, args.min_count, db))
    elif args.command == "export":
        print(export_transactions(args.wallet, args.format, db))
    elif args.command == "list":
        _print_json(list_payments(args.wallet, args.coin, args.status, args.limit, db))
    elif args.command == "register-wallet":
        _print_json(register_wallet(args.address, args.coin, args.label, db))
    elif args.command == "network-stats":
        print(json.dumps(network_stats(db), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
