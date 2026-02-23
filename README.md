# blackroad-crypto-payment-processor

Production-grade crypto payment processing for BTC, ETH, SOL, USDC, MATIC, and AVAX.

## Features
- Multi-coin payment creation with auto-generated mock tx hashes
- Mock confirmation engine based on realistic block times per coin
- USD value calculation using configurable exchange rates
- Network fee estimation per coin type
- Wallet summary with net balance per coin and USD total
- Fraud detection: high-value threshold alerts and rapid-payment velocity checks
- Transaction export in JSON and CSV formats
- SQLite persistence with WAL mode

## Supported Coins
| Coin | Block Time | Required Confirmations |
|------|-----------|------------------------|
| BTC  | ~10 min   | 3                      |
| ETH  | ~15 sec   | 12                     |
| SOL  | ~2 sec    | 32                     |
| USDC | ~15 sec   | 12                     |
| MATIC| ~3 sec    | 64                     |
| AVAX | ~2 sec    | 10                     |

## Usage
```bash
python crypto_payments.py init
python crypto_payments.py create 0xSEND 0xRECV 1.5 ETH --memo "Payment for services"
python crypto_payments.py status <payment_id>
python crypto_payments.py wallet-summary 0xSEND
python crypto_payments.py suspicious --threshold 50000
python crypto_payments.py export --format csv --wallet 0xSEND
python crypto_payments.py network-stats
```

## Testing
```bash
pip install pytest
pytest test_crypto_payments.py -v
```
