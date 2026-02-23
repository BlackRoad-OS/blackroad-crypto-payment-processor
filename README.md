# BlackRoad Crypto Payment Processor

> Invoice management for crypto payments: BTC, ETH, USDC, SOL and more — with exchange rate simulation, confirmation tracking, and CSV accounting export.

Part of the [BlackRoad OS](https://github.com/BlackRoad-OS) platform.

## Features

- **Invoice lifecycle**: `create_invoice()` → `mark_confirmed()` → accounting export
- **Multi-currency**: USD, EUR, GBP, JPY → BTC, ETH, USDC, USDT, SOL, LTC, XRP
- **Exchange rates**: Mock rates (CoinGecko-compatible interface for live data)
- **Confirmation tracking**: Per-crypto required confirmations (BTC=6, ETH=12, SOL=32)
- **Accounting CSV**: Export confirmed invoices for bookkeeping
- **Revenue summary**: Aggregated by fiat currency and crypto asset

## Usage

```bash
# Create invoice
python src/crypto_payment_processor.py create 100.00 --currency USD --crypto BTC

# Mark confirmed
python src/crypto_payment_processor.py confirm INV-XXXXXXXX <txhash> --confirmations 6

# List pending
python src/crypto_payment_processor.py pending

# Export accounting CSV
python src/crypto_payment_processor.py export

# Get exchange rate
python src/crypto_payment_processor.py rate USD BTC
```

## Architecture

- `src/crypto_payment_processor.py` — 660+ lines: `Invoice`, `Confirmation`, `CryptoPaymentDB`, `CryptoPaymentProcessor`
- `tests/` — 18 test functions
- SQLite: `invoices` + `confirmations` tables

## License

Proprietary — © BlackRoad OS, Inc. All rights reserved.
