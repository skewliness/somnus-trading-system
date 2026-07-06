# Gann Structural Feature — Test Results & Recommendation
# Gann 结构特征 — 测试结果与建议

> CONFIDENTIAL / 保密 — Somnus 项目内部文档。

## Changelog / 变更日志

| 时间 / Time | 作者 / Author | 变更 / Change |
|---|---|---|
| 2026-07-07 | Fable5 (Sean's session) | Initial report: cross-symbol stratification (live data), portfolio A/B pipeline, recommendation. 首版。 |

---

## TL;DR

**Gann must NOT become a global hard gate. It qualifies as a BTC-only macro position-sizing factor.**
**Gann 不能做全局硬门槛；它只够格做 BTC 单币种的宏观仓位因子。**

By the task request's own criterion — "If this ranking is not stable across BTC/ETH/SOL/BNB/HYPE/ZEC, Gann should not become a global hard gate" — the ranking `above_1x1 > 1x2_to_1x1 > below_1x2` holds **only on BTC** and fails on every other symbol tested.

## 1. Cross-symbol stratification / 跨币种分层（真实数据）

Data: Binance USDS futures daily klines, 2021-01-01 → 2026-07-06, walk-forward
`compute_gann_feature(asof_index=i)` (no future data in anchors; unit-tested).
Script: `scripts/evaluate_gann_structural_features.py`; row-level output 7,050 samples at
`reports/gann_eval/gann_zone_stratification_rows.csv`.

| Symbol | 7d | 14d | 30d | Verdict |
|---|---|---|---|---|
| BTCUSDT | HOLDS | HOLDS | HOLDS | **HOLDS** |
| ETHUSDT | FAILS | FAILS | FAILS | FAILS |
| SOLUSDT | FAILS | HOLDS | HOLDS | FAILS |
| BNBUSDT | FAILS | HOLDS | HOLDS | FAILS |
| HYPEUSDT | n/a | n/a | n/a | NO DATA（上市晚，365d 预热后无有效样本） |
| ZECUSDT | FAILS | FAILS | FAILS | FAILS |

**ZEC 反转证据 / ZEC rebound evidence**（与 2026-07-06 任务书中的怀疑一致，现已量化）：
ZEC `below_1x2` 30d：n=641，avg **+19.30%**，PF **3.18**，avg MFE 42.1%。
`below_1x2` 在 ZEC 上是强反弹区，机械解读为空头延续区会系统性做反方向。
ETH 同样反常：`above_1x1` 7/14/30d 平均收益全部为负（PF 0.82–0.93），中间区 `1x2_to_1x1` 最好（PF 1.19–1.65）。

## 2. Portfolio A/B / 组合级 A/B

Script: `scripts/backtest_v6_gann_ab.py` — next-bar 成交、fee 5bps + slippage 3bps（单边）、
`funding_omitted=true` 显式标注、组合约束（max_concurrent=5、per-symbol=1、同日按分数分配、
重叠信号忽略并计数）、OOS 切分 train ≤2024-12-31 / validation ≥2025-01-01。

**生产候选模式**（等待 xuejian 提供 V6 replay 候选导出）：
`python3 scripts/backtest_v6_gann_ab.py --candidates-file v6_candidates.jsonl --asof <date>`
JSONL schema 见脚本 docstring。

**Demo 模式结果**（20d SMA cross 候选，n=1,121，明确标注非 V6 生产信号；管道验证 + 方向性参考）：

| Arm | Train return / PF | Validation return / PF | Trades (t/v) |
|---|---|---|---|
| A0 baseline | +10.20% / 1.0162 | -17.63% / 0.905 | 694 / 262 |
| A1 low-weight scoring | 与 A0 完全相同 | 与 A0 完全相同 | 694 / 262 |
| A2 position sizing | **+11.87% / 1.0205** | **-16.72% / 0.912** | 694 / 262 |
| A3 hard filter | +3.86% / 1.0027 | -16.70% / 0.894 | 554 / 202 |

与原 n=2 表的结论方向一致，现在样本量 n=694/262：
- A1 弱评分再次对行为零影响（穿不透分配层）
- **A2 仓位乘数在两个窗口都改善收益与 PF，交易数不变** —— 唯一双窗口稳健的臂
- A3 硬过滤 train 大幅劣化（漏掉 140 笔），validation 无优势 —— 否决

## 3. Recommendation / 建议

1. **启用范围**：`gann_position_sizing_enabled` 仅对 BTCUSDT 开启（A2 模式，×1.10–1.25 / ×0.50–0.75）。
2. **ZEC 特例**：`below_1x2` 在 ZEC 上 bias 强制 neutral 或按反弹逻辑处理，禁止作为空头依据。
3. **ETH/SOL/BNB**：不启用，直到出现跨期稳定证据；可保留解释性输出（confidence=0）。
4. **HYPE**：数据不足，不启用。
5. **全局硬门槛：否决**（跨币种排序不稳定 + A3 双窗口劣化）。
6. **下一步**：用真实 V6 replay 候选跑 `--candidates-file` 模式，替换 demo 数字后再定最终乘数区间。

## 4. Reproduce / 复现

```bash
python3 scripts/evaluate_gann_structural_features.py --start-date 2021-01-01
python3 scripts/backtest_v6_gann_ab.py --demo --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ZECUSDT --asof 2026-07-05 --offline
python3 -m pytest tests/test_gann_feature.py tests/test_evaluate_gann_helpers.py tests/test_backtest_v6_gann_ab.py -q
```
