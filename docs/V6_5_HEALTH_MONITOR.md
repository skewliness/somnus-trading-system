# V6.5 Runtime Health Monitor

The health monitor is read-only and does not participate in strategy or order
decisions. It checks:

- execution-service status and summary freshness;
- market-data freshness already attached to signals;
- latency and failures observed by existing Binance requests;
- whether ledger reconciliation covers the latest position mutation;
- whether submitted exchange positions retain a protective stop.

It does not issue a probe request, increase polling frequency, modify a
position, place an order, or change a strategy threshold. The Telegram monitor
writes a compact health report at most once per minute (or immediately when a
critical finding identity changes) and sends only new critical findings through
the existing deduplicated alert path. Existing service-down and stale-summary
alerts remain authoritative and are not duplicated.
