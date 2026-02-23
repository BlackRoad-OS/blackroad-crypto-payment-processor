"""
BlackRoad Crypto Payment Processor
====================================
Production-quality crypto invoice management with exchange rate
simulation, SQLite persistence, and CSV accounting export.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

# ─── Configuration ────────────────────────────────────────────────────────────
DB_PATH = Path.home() / ".blackroad" / "crypto_payments.db"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger("crypto_payments")

PRECISION_FIAT = Decimal("0.01")
PRECISION_CRYPTO = Decimal("0.00000001")  # 8 decimal places (Satoshi)

# Mock exchange rates (base: USD). In production, fetch from CoinGecko etc.
MOCK_RATES_TO_USD: Dict[str, Decimal] = {
    "USD": Decimal("1.00"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.26"),
    "JPY": Decimal("0.0067"),
    "CAD": Decimal("0.74"),
    "BTC": Decimal("67500.00"),
    "ETH": Decimal("3500.00"),
    "USDC": Decimal("1.00"),
    "USDT": Decimal("1.00"),
    "SOL": Decimal("155.00"),
    "LTC": Decimal("82.00"),
    "XRP": Decimal("0.58"),
}

CRYPTO_ADDRESS_PREFIXES: Dict[str, str] = {
    "BTC": "bc1q",
    "ETH": "0x",
    "USDC": "0x",
    "USDT": "0x",
    "SOL": "",
    "LTC": "ltc1q",
    "XRP": "r",
}

CONFIRMATIONS_REQUIRED: Dict[str, int] = {
    "BTC": 6,
    "ETH": 12,
    "USDC": 12,
    "USDT": 12,
    "SOL": 32,
    "LTC": 6,
    "XRP": 1,
}


# ─── Enumerations ─────────────────────────────────────────────────────────────
class InvoiceStatus(str, Enum):
    PENDING = "pending"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    OVERPAID = "overpaid"


# ─── Data Classes ─────────────────────────────────────────────────────────────
@dataclass
class Invoice:
    id: str
    amount_fiat: Decimal
    currency: str
    crypto: str
    crypto_amount: Decimal
    address: str
    status: InvoiceStatus
    exchange_rate: Decimal
    description: str = ""
    tx_hash: Optional[str] = None
    confirmations: int = 0
    confirmations_required: int = 6
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    metadata: str = ""

    def __post_init__(self):
        for attr in ("amount_fiat", "crypto_amount", "exchange_rate"):
            val = getattr(self, attr)
            if isinstance(val, (int, float, str)):
                setattr(self, attr, Decimal(str(val)))
        if isinstance(self.status, str):
            self.status = InvoiceStatus(self.status)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.expires_at, str) and self.expires_at:
            self.expires_at = datetime.fromisoformat(self.expires_at)
        if isinstance(self.confirmed_at, str) and self.confirmed_at:
            self.confirmed_at = datetime.fromisoformat(self.confirmed_at)

    @property
    def is_expired(self) -> bool:
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return self.status == InvoiceStatus.PENDING
        return False

    @property
    def fiat_display(self) -> str:
        symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CAD": "CA$"}
        sym = symbols.get(self.currency, self.currency + " ")
        return f"{sym}{self.amount_fiat}"


@dataclass
class Confirmation:
    id: str
    invoice_id: str
    tx_hash: str
    block_height: Optional[int]
    confirmations: int
    confirmed: bool
    recorded_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if isinstance(self.recorded_at, str):
            self.recorded_at = datetime.fromisoformat(self.recorded_at)


@dataclass
class ExchangeRate:
    from_currency: str
    to_currency: str
    rate: Decimal
    source: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)


# ─── Database Layer ────────────────────────────────────────────────────────────
class CryptoPaymentDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self.transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS invoices (
                    id                     TEXT PRIMARY KEY,
                    amount_fiat            TEXT NOT NULL,
                    currency               TEXT NOT NULL,
                    crypto                 TEXT NOT NULL,
                    crypto_amount          TEXT NOT NULL,
                    address                TEXT NOT NULL,
                    status                 TEXT NOT NULL DEFAULT 'pending',
                    exchange_rate          TEXT NOT NULL,
                    description            TEXT NOT NULL DEFAULT '',
                    tx_hash                TEXT,
                    confirmations          INTEGER NOT NULL DEFAULT 0,
                    confirmations_required INTEGER NOT NULL DEFAULT 6,
                    created_at             TEXT NOT NULL,
                    expires_at             TEXT,
                    confirmed_at           TEXT,
                    metadata               TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS confirmations (
                    id           TEXT PRIMARY KEY,
                    invoice_id   TEXT NOT NULL REFERENCES invoices(id),
                    tx_hash      TEXT NOT NULL,
                    block_height INTEGER,
                    confirmations INTEGER NOT NULL DEFAULT 0,
                    confirmed    INTEGER NOT NULL DEFAULT 0,
                    recorded_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_invoices_status
                    ON invoices(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_invoices_address
                    ON invoices(address);
                CREATE INDEX IF NOT EXISTS idx_confirmations_invoice
                    ON confirmations(invoice_id);
            """)

    def save_invoice(self, invoice: Invoice) -> Invoice:
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO invoices
                   (id, amount_fiat, currency, crypto, crypto_amount, address,
                    status, exchange_rate, description, tx_hash, confirmations,
                    confirmations_required, created_at, expires_at, confirmed_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    invoice.id, str(invoice.amount_fiat), invoice.currency,
                    invoice.crypto, str(invoice.crypto_amount), invoice.address,
                    invoice.status.value, str(invoice.exchange_rate),
                    invoice.description, invoice.tx_hash, invoice.confirmations,
                    invoice.confirmations_required,
                    invoice.created_at.isoformat(),
                    invoice.expires_at.isoformat() if invoice.expires_at else None,
                    invoice.confirmed_at.isoformat() if invoice.confirmed_at else None,
                    invoice.metadata,
                ),
            )
        return invoice

    def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM invoices WHERE id=?", (invoice_id,)
            ).fetchone()
            return self._row_to_invoice(row) if row else None
        finally:
            conn.close()

    def update_invoice_status(
        self,
        invoice_id: str,
        status: InvoiceStatus,
        tx_hash: Optional[str] = None,
        confirmations: Optional[int] = None,
        confirmed_at: Optional[datetime] = None,
    ):
        with self.transaction() as conn:
            updates: list = [f"status='{status.value}'"]
            params: list = []
            if tx_hash is not None:
                updates.append("tx_hash=?")
                params.append(tx_hash)
            if confirmations is not None:
                updates.append("confirmations=?")
                params.append(confirmations)
            if confirmed_at is not None:
                updates.append("confirmed_at=?")
                params.append(confirmed_at.isoformat())
            params.append(invoice_id)
            conn.execute(
                f"UPDATE invoices SET {', '.join(updates)} WHERE id=?", params
            )

    def save_confirmation(self, conf: Confirmation):
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO confirmations
                   (id, invoice_id, tx_hash, block_height, confirmations, confirmed, recorded_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    conf.id, conf.invoice_id, conf.tx_hash,
                    conf.block_height, conf.confirmations,
                    1 if conf.confirmed else 0,
                    conf.recorded_at.isoformat(),
                ),
            )

    def list_invoices(
        self,
        status: Optional[InvoiceStatus] = None,
        crypto: Optional[str] = None,
        limit: int = 100,
    ) -> List[Invoice]:
        conn = self._connect()
        try:
            query = "SELECT * FROM invoices WHERE 1=1"
            params: list = []
            if status:
                query += " AND status=?"
                params.append(status.value)
            if crypto:
                query += " AND crypto=?"
                params.append(crypto.upper())
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_invoice(r) for r in rows]
        finally:
            conn.close()

    @staticmethod
    def _row_to_invoice(row: sqlite3.Row) -> Invoice:
        return Invoice(
            id=row["id"],
            amount_fiat=Decimal(row["amount_fiat"]),
            currency=row["currency"],
            crypto=row["crypto"],
            crypto_amount=Decimal(row["crypto_amount"]),
            address=row["address"],
            status=InvoiceStatus(row["status"]),
            exchange_rate=Decimal(row["exchange_rate"]),
            description=row["description"] or "",
            tx_hash=row["tx_hash"],
            confirmations=row["confirmations"],
            confirmations_required=row["confirmations_required"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            confirmed_at=datetime.fromisoformat(row["confirmed_at"]) if row["confirmed_at"] else None,
            metadata=row["metadata"] or "",
        )


# ─── Payment Processor Service ─────────────────────────────────────────────────
class CryptoPaymentProcessor:
    """Core crypto payment processing service."""

    def __init__(self, db: Optional[CryptoPaymentDB] = None):
        self.db = db or CryptoPaymentDB()

    # ── Exchange Rate ────────────────────────────────────────────────────────

    def calculate_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
        mock: bool = True,
    ) -> ExchangeRate:
        """Calculate exchange rate between two currencies."""
        from_c = from_currency.upper()
        to_c = to_currency.upper()

        if not mock:
            raise NotImplementedError(
                "Live exchange rates require API key configuration. "
                "Set mock=True or configure COINGECKO_API_KEY."
            )

        if from_c not in MOCK_RATES_TO_USD:
            raise ValueError(f"Unsupported currency: {from_c}")
        if to_c not in MOCK_RATES_TO_USD:
            raise ValueError(f"Unsupported currency: {to_c}")

        from_usd = MOCK_RATES_TO_USD[from_c]
        to_usd = MOCK_RATES_TO_USD[to_c]
        rate = (from_usd / to_usd).quantize(PRECISION_CRYPTO, rounding=ROUND_HALF_UP)

        return ExchangeRate(
            from_currency=from_c,
            to_currency=to_c,
            rate=rate,
            source="mock" if mock else "coingecko",
        )

    def _generate_address(self, crypto: str) -> str:
        """Generate a deterministic mock payment address."""
        prefix = CRYPTO_ADDRESS_PREFIXES.get(crypto.upper(), "")
        suffix = uuid.uuid4().hex[:32]
        return f"{prefix}{suffix}"

    def _fiat_to_crypto(
        self, amount_fiat: Decimal, currency: str, crypto: str
    ) -> Tuple[Decimal, Decimal]:
        """Convert fiat amount to crypto. Returns (crypto_amount, rate)."""
        rate_obj = self.calculate_exchange_rate(currency, crypto)
        crypto_amount = (amount_fiat * rate_obj.rate).quantize(
            PRECISION_CRYPTO, rounding=ROUND_HALF_UP
        )
        return crypto_amount, rate_obj.rate

    # ── Invoice Operations ───────────────────────────────────────────────────

    def create_invoice(
        self,
        amount: Decimal,
        currency: str = "USD",
        crypto: str = "BTC",
        description: str = "",
        expires_minutes: int = 60,
        metadata: str = "",
    ) -> Invoice:
        """Create a new crypto payment invoice."""
        amount = Decimal(str(amount)).quantize(PRECISION_FIAT, rounding=ROUND_HALF_UP)
        if amount <= Decimal("0"):
            raise ValueError("Invoice amount must be positive")

        crypto_upper = crypto.upper()
        currency_upper = currency.upper()

        crypto_amount, exchange_rate = self._fiat_to_crypto(amount, currency_upper, crypto_upper)
        address = self._generate_address(crypto_upper)
        conf_required = CONFIRMATIONS_REQUIRED.get(crypto_upper, 6)

        from datetime import timedelta
        invoice = Invoice(
            id=f"INV-{str(uuid.uuid4()).upper()[:8]}",
            amount_fiat=amount,
            currency=currency_upper,
            crypto=crypto_upper,
            crypto_amount=crypto_amount,
            address=address,
            status=InvoiceStatus.PENDING,
            exchange_rate=exchange_rate,
            description=description,
            confirmations_required=conf_required,
            expires_at=datetime.utcnow() + timedelta(minutes=expires_minutes),
            metadata=metadata,
        )
        self.db.save_invoice(invoice)
        logger.info(
            "Invoice created: %s | %s → %s %s @ %s",
            invoice.id, invoice.fiat_display, crypto_amount, crypto_upper, exchange_rate,
        )
        return invoice

    def mark_confirmed(
        self,
        invoice_id: str,
        tx_hash: str,
        block_height: Optional[int] = None,
        confirmations: int = 1,
    ) -> Invoice:
        """Mark invoice as confirmed with transaction hash."""
        invoice = self.db.get_invoice(invoice_id)
        if not invoice:
            raise ValueError(f"Invoice {invoice_id} not found")
        if invoice.status == InvoiceStatus.CONFIRMED:
            raise ValueError(f"Invoice {invoice_id} already confirmed")
        if invoice.status == InvoiceStatus.CANCELLED:
            raise ValueError(f"Invoice {invoice_id} is cancelled")

        fully_confirmed = confirmations >= invoice.confirmations_required
        new_status = InvoiceStatus.CONFIRMED if fully_confirmed else InvoiceStatus.AWAITING_CONFIRMATION
        confirmed_at = datetime.utcnow() if fully_confirmed else None

        self.db.update_invoice_status(
            invoice_id, new_status,
            tx_hash=tx_hash,
            confirmations=confirmations,
            confirmed_at=confirmed_at,
        )

        conf = Confirmation(
            id=str(uuid.uuid4()),
            invoice_id=invoice_id,
            tx_hash=tx_hash,
            block_height=block_height,
            confirmations=confirmations,
            confirmed=fully_confirmed,
        )
        self.db.save_confirmation(conf)

        logger.info(
            "Invoice %s: %s (txhash=%s, confs=%d/%d)",
            invoice_id, new_status.value, tx_hash[:12], confirmations, invoice.confirmations_required
        )
        return self.db.get_invoice(invoice_id)

    def cancel_invoice(self, invoice_id: str) -> Invoice:
        """Cancel a pending invoice."""
        invoice = self.db.get_invoice(invoice_id)
        if not invoice:
            raise ValueError(f"Invoice {invoice_id} not found")
        if invoice.status not in (InvoiceStatus.PENDING, InvoiceStatus.AWAITING_CONFIRMATION):
            raise ValueError(f"Cannot cancel invoice with status: {invoice.status.value}")
        self.db.update_invoice_status(invoice_id, InvoiceStatus.CANCELLED)
        return self.db.get_invoice(invoice_id)

    def list_pending_invoices(self) -> List[Invoice]:
        """Return all pending invoices."""
        return self.db.list_invoices(status=InvoiceStatus.PENDING)

    def list_invoices(
        self,
        status: Optional[str] = None,
        crypto: Optional[str] = None,
    ) -> List[Invoice]:
        """List invoices with optional filtering."""
        status_enum = InvoiceStatus(status) if status else None
        return self.db.list_invoices(status=status_enum, crypto=crypto)

    # ── Accounting ───────────────────────────────────────────────────────────

    def export_accounting_csv(
        self,
        include_statuses: Optional[List[InvoiceStatus]] = None,
    ) -> str:
        """Export invoices as CSV for accounting purposes."""
        if include_statuses is None:
            include_statuses = [InvoiceStatus.CONFIRMED]

        all_invoices: List[Invoice] = []
        for status in include_statuses:
            all_invoices.extend(self.db.list_invoices(status=status, limit=10000))

        all_invoices.sort(key=lambda i: i.created_at)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Invoice ID", "Created At", "Confirmed At", "Status",
            "Fiat Amount", "Currency", "Crypto", "Crypto Amount",
            "Exchange Rate", "Address", "TX Hash", "Confirmations",
            "Description",
        ])
        for inv in all_invoices:
            writer.writerow([
                inv.id, inv.created_at.isoformat(),
                inv.confirmed_at.isoformat() if inv.confirmed_at else "",
                inv.status.value,
                str(inv.amount_fiat), inv.currency,
                inv.crypto, str(inv.crypto_amount),
                str(inv.exchange_rate), inv.address,
                inv.tx_hash or "", inv.confirmations,
                inv.description,
            ])
        return output.getvalue()

    def revenue_summary(self) -> Dict:
        """Summarize confirmed revenue by currency."""
        confirmed = self.db.list_invoices(status=InvoiceStatus.CONFIRMED, limit=100000)
        summary: Dict[str, Decimal] = {}
        by_crypto: Dict[str, Decimal] = {}
        for inv in confirmed:
            summary[inv.currency] = summary.get(inv.currency, Decimal("0")) + inv.amount_fiat
            by_crypto[inv.crypto] = by_crypto.get(inv.crypto, Decimal("0")) + inv.crypto_amount
        return {
            "total_invoices": len(confirmed),
            "by_fiat_currency": {k: str(v) for k, v in summary.items()},
            "by_crypto": {k: str(v) for k, v in by_crypto.items()},
        }


# ─── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto-payments",
        description="BlackRoad Crypto Payment Processor CLI",
    )
    parser.add_argument("--db", default=str(DB_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="Create invoice")
    p.add_argument("amount")
    p.add_argument("--currency", default="USD")
    p.add_argument("--crypto", default="BTC")
    p.add_argument("--desc", default="")
    p.add_argument("--expires", type=int, default=60)

    p = sub.add_parser("confirm", help="Mark invoice confirmed")
    p.add_argument("invoice_id")
    p.add_argument("txhash")
    p.add_argument("--confirmations", type=int, default=6)
    p.add_argument("--block", type=int, default=None)

    p = sub.add_parser("cancel", help="Cancel pending invoice")
    p.add_argument("invoice_id")

    p = sub.add_parser("list", help="List invoices")
    p.add_argument("--status", default=None)
    p.add_argument("--crypto", default=None)

    sub.add_parser("pending", help="List pending invoices")

    p = sub.add_parser("export", help="Export accounting CSV")
    p.add_argument("--all", action="store_true")

    p = sub.add_parser("rate", help="Get exchange rate")
    p.add_argument("from_currency")
    p.add_argument("to_currency")

    sub.add_parser("summary", help="Revenue summary")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    processor = CryptoPaymentProcessor(CryptoPaymentDB(Path(args.db)))

    if args.command == "create":
        inv = processor.create_invoice(
            amount=Decimal(args.amount),
            currency=args.currency,
            crypto=args.crypto,
            description=args.desc,
            expires_minutes=args.expires,
        )
        print(f"✓ Invoice created: {inv.id}")
        print(f"  Amount:  {inv.fiat_display} ({inv.currency})")
        print(f"  Crypto:  {inv.crypto_amount} {inv.crypto}")
        print(f"  Address: {inv.address}")
        print(f"  Rate:    1 {inv.crypto} = {1/inv.exchange_rate:.2f} {inv.currency}")
        print(f"  Expires: {inv.expires_at}")

    elif args.command == "confirm":
        inv = processor.mark_confirmed(
            args.invoice_id, args.txhash,
            confirmations=args.confirmations,
            block_height=args.block,
        )
        print(f"✓ Invoice {inv.id}: {inv.status.value}")

    elif args.command == "cancel":
        inv = processor.cancel_invoice(args.invoice_id)
        print(f"✓ Invoice {inv.id}: {inv.status.value}")

    elif args.command == "list":
        invoices = processor.list_invoices(args.status, args.crypto)
        for inv in invoices:
            print(f"  {inv.id}  {inv.status.value:<22}  {inv.fiat_display:>10}  {inv.crypto_amount} {inv.crypto}")

    elif args.command == "pending":
        invoices = processor.list_pending_invoices()
        print(f"Pending invoices: {len(invoices)}")
        for inv in invoices:
            print(f"  {inv.id}  {inv.fiat_display}  {inv.crypto_amount} {inv.crypto}  expires={inv.expires_at}")

    elif args.command == "export":
        statuses = None
        if args.all:
            statuses = list(InvoiceStatus)
        print(processor.export_accounting_csv(statuses))

    elif args.command == "rate":
        rate = processor.calculate_exchange_rate(args.from_currency, args.to_currency)
        print(f"1 {rate.from_currency} = {rate.rate} {rate.to_currency} (source: {rate.source})")

    elif args.command == "summary":
        s = processor.revenue_summary()
        print(f"Total confirmed invoices: {s['total_invoices']}")
        for cur, amt in s['by_fiat_currency'].items():
            print(f"  {cur}: {amt}")
        for cry, amt in s['by_crypto'].items():
            print(f"  {cry}: {amt}")


if __name__ == "__main__":
    main()
