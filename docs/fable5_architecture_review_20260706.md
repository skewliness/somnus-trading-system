# Somnus V5/V6 Architecture Refactor Review — Fable5 Deliverable
# Somnus V5/V6 架构重构评审 — Fable5 交付物

> **CONFIDENTIAL / 保密** — 仅供文档所有者（skewliness/xuejian）与授权协作方（Sean）使用。请勿外传。
> For the document owner and authorized collaborators only. Do not redistribute.

## Changelog / 变更日志

| 时间 / Time | 作者 / Author | 变更 / Change |
|---|---|---|
| 2026-07-06 | Fable5 (Sean's session) | Initial full review: architecture map, 63-gate matrix, Gann design, A/B plan, TG templates, staged rollout, risk checklist. 首版完整评审。 |

---

## TL;DR (EN)

**Verdict: the system is structurally over-gated — confirmed from code, not just from replay numbers.** The V6 entry path is a serial chain of ~39 hard gates (all must pass; any one drops the candidate entirely), concentrated in `trading_system/v6_execution.py:5690–6674` plus market-level blocks at `:7690–8599`. This matches the cited replay result (2,752 candidates → 14 executed, 0.51% coverage — *cited from the review-plan PDF, not independently verifiable from this snapshot*). However, the force-entry arm (54.55% win rate) is **not** directly promotable: it ignores portfolio constraints, stacking, funding, slippage, and symbol risk profiles. The correct fix is re-classifying gates into hard-gate / soft-score / sizing / risk-exit, exactly as the brief proposes. Gann code is **entirely absent from this snapshot** — the Gann section below is a greenfield design spec. The Telegram role-split target (V5/V6 bots = execution+errors only) is **not yet implemented**: the current monitor still routes `p/fc/report/news` commands (`scripts/telegram_monitor_v5_v6.py:5060–5177`).

**评审结论：系统在代码结构层面确认过度门控**——V6 入场是约 39 个硬门槛的串行链，任何一关失败候选即被丢弃。修复方向不是放开全部门槛（force-entry 不可直接上线），而是把门槛重新分类为「硬门槛 / 软评分 / 仓位调节 / 风险退出」四类。

### Evidence caveats / 证据边界（必读）

1. **评审对象**：GitHub 快照 `skewliness/somnus-trading-system@56d1816`（code-only export）。生产库 `/Users/ma7/binance-trading-skill-system` 与 `E:\somnus\output` 证据文件**不在本机**。
2. **Replay 数字均引自评审任务书 PDF，未独立复核**（2,752/14/0.51%、force-entry 1,287/54.55%、Gann A/B 表）。文中以 *[PDF-cited]* 标注。
3. **快照不可运行**：导出删除了 `trading_system/smart_money_panel` 与 `trading_system/coin_scout_4h_labeler`，但 `trading_system/execution.py:15` 等处顶层 import 仍在 → 核心执行模块 import 即失败。pytest：252 通过 / 23 失败（全部由缺失模块级联 + 1 个 `macro_trend` 断言差异）。
4. **Gann 代码在快照中完全不存在**（全库 grep "gann" / "1x1" / "江恩" 零命中；`v6_gann_ab_backtest.py`、`gann_nextbar_portfolio_backtest.py` 不在快照内）。

---

## 1. Architecture Map / 架构图

### 1.1 Layer diagram / 分层

```
┌─ DATA 数据层 ──────────────────────────────────────────────────┐
│ binance.py (BinanceFuturesClient + KlineCache)                 │
│ config.py (StrategyConfig, 240+ env params)                    │
│ persistence.py (fcntl file-lock + atomic JSON)                 │
└──────────────────────────┬─────────────────────────────────────┘
┌─ SIGNAL 信号层 ───────────▼─────────────────────────────────────┐
│ bb_structure.py  (BB 96/105/120/210/300/672/1440/…/4707)       │
│ h4_structure.py  (4H swing/MA20/ATR)   kline_pattern.py        │
│ oi_divergence_system.py  macro_trend.py  v5_derivatives_flow   │
│ sentiment.py + bridges.py (manual/Square/CoinGecko/OpenNews/   │
│   OpenTwitter/AI-Select/skill_bridges)  meme_radar.py          │
│ news_wss_listener.py (6551 WSS → opennews_wss.json)            │
│ coin_scout_model.py (LightGBM P(direction valid))              │
│ hf_signal_cache.py (回测质量 TTL 缓存)                           │
└──────────────────────────┬─────────────────────────────────────┘
┌─ STRATEGY 策略/评分层 ─────▼─────────────────────────────────────┐
│ scoring.py (100分制加权)  regime.py  candidates.py (basket)     │
│ parameter_optimizer.py (Bayesian TP/SL search)                 │
└──────────────────────────┬─────────────────────────────────────┘
┌─ EXECUTION 执行层 ─────────▼────────────────────────────────────┐
│ V5: execution.py (PaperExecutionEngine) → execution_bridge.py  │
│     → execution_runner.py → Binance REST                       │
│ V6: v6_execution.py (605KB 单体; 门槛链 + 下单 + 风控 + 账本)      │
│ overlay: scripts/run_v5_v6_execution.py (V5候选 → V6执行)        │
│ v5_loss_prevention.py (report-only 风险标签)                    │
└──────────────────────────┬─────────────────────────────────────┘
┌─ LEDGER 账本 ──────────────▼────────────────────────────────────┐
│ V5: data/trading_system/paper_state.json                       │
│ V6: data/trading_system/v6_execution/  (完全分离, 无自动同步)      │
└──────────────────────────┬─────────────────────────────────────┘
┌─ BROADCAST 播报 ───────────▼────────────────────────────────────┐
│ scripts/telegram_monitor_v5_v6.py (执行/告警/摘要/命令路由)        │
│ scripts/xuejian_proactive_push.py (地缘+9点宏观)                  │
│ scripts/hermes_somnus_command.py (p/fc/report/news 交互)         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 V5 / V6 / Xuejian boundary / 三系统边界

| 维度 | V5 | V6 | Xuejian/Hermes |
|---|---|---|---|
| 定位 | 小/中市值、热点币、异动扫描 | 主流币策略（BTC/ETH/SOL/BNB/HYPE/ZEC） | 情报、宏观、解释，**无执行能力** |
| 信号入口 | pipeline.py:103+ 六阶段编排 | 同一 pipeline 候选 + 自有门槛链 | 读 opennews_wss.json 等桥文件 |
| 执行 | execution.py:141+ → bridge → runner | v6_execution.py 直接下单 | 无 |
| 账本 | paper_state.json | v6_execution/ 独立目录 | 只读 |
| 混合模式 | `TS_V5_USE_V6_EXECUTION_OVERLAY=1`（run_pipeline_24h.sh:25）时 V5 候选喂 V6 执行 | ← | — |

### 1.3 Data flow (one candidate) / 单候选全链路

`pipeline.py:run()` 情绪聚合(:126-137) → `engine.py:scan()`(:24-32) 收集 MarketSnapshot → `scoring.py:score_snapshot()`(:31-150) 打分 → `candidates.py:build_candidate_basket()`(:34+) 分级(A/B/C, executable_now/confirm_needed/watch_only) → **V6 门槛链**（§2） → `execution_bridge.py:write_requests()`(:37-86) 生成 intent JSON → `execution_runner.py:execute_request_file()`(:49-88) 签名提交 `/fapi/v1/order` → 结果落 `execution_results/` → 账本更新 → `telegram_monitor_v5_v6.py:format_execution_intent()`(:4751) 播报。

---

## 2. Gate Classification Matrix / 门槛分类矩阵

代码盘点共 **63 个拦截/过滤条件**：39 硬门槛（候选直接丢弃）、8 个 watchable 观察项、16 个软修饰（评分/标签）。PDF 列出的主要 rejection reason 全部在代码中定位。

**分类原则（按任务书）**：硬门槛只保留「不可交易/数据不可靠/账实不符/用户手动禁/交易所异常未对账」五类；其余降级为软评分、仓位调节或风险退出。

### 2.1 建议保留为 HARD 的门槛（8 个）

| Gate | 位置 | 理由 |
|---|---|---|
| `entry_blocked_stale_market_data` | v6_execution.py:7690 | 数据不可靠 = 不交易，无争议 |
| `technical_entry_only_no_indicator_level` | :5907 | 无任何入场参考位 = 无法定义止损，保留；但应**扩充 level 菜单**（当前 symbol profile 白名单过窄放大了它的触发率） |
| `setup_type_not_allowed` / `setup_type_no_edge` | execution.py:984/:983 | 无正期望 setup 不入场 |
| `short_blocked_by_smart_signal_short_squeeze` / `long_blocked_by_smart_signal_long_crowding` | v6_execution.py:7707/7710 | 挤仓结构性风险，非方向分歧 |
| `short_blocked_market_daily_ma20_reclaim` / `long_blocked_daily_ma20_loss` | :8543/:8599 | 日线论点失效 = 论点级止损，属风险退出范畴，保留 |
| `consecutive_entry_blocked` | execution.py:1464 | 防连损，实为风险预算，保留（或改为冷却+减半仓） |

### 2.2 建议降级为 SOFT SCORE / SIZING 的门槛（重点 12 个）

| Gate | 位置 | 现状 | 建议 |
|---|---|---|---|
| `macro_direction_mismatch` | v6_execution.py:5707 | HARD | → **sizing 0.5×** + 要求 15M 三线确认；仅在「reversal scout」模式下允许小仓。这是 PDF 列出的头号拒绝原因，也是覆盖率杀手 |
| `macro_direction_neutral` | :5690 | HARD | → 不拦，中性即不加不减（1.0×） |
| `long_blocked_15m_ma20_downtrend_alive` | :5975 | HARD | → 确认型软门槛：等 1 根 15M 收线确认反转，或 0.5× 入场 |
| `short_blocked_15m_ma20_trend_alive` | :5972 | HARD | 同上 |
| `short_15m_blocked_by_4h_right_side_long` | :6190 | HARD | → 普通 15M 空单默认拦，但允许「momentum chase」模式带 3 线确认 + 0.3× 小仓 + 快止损 |
| `symbol_bb_entry_profile_selected_level_blocked` | :6277 | HARD | → 评分惩罚（-15）。profile 来自历史回测，样本会漂移，不应永久硬禁 |
| `symbol_bb_entry_profile_selected_level_not_allowed` | :6298 | HARD | → 同上，白名单改成先验分数 |
| `trend_big_blocked_without_4h_right_side` | :6003 | HARD | → trend 单保留确认要求，但开放「structural pullback」模式在 BB 支撑处提前小仓 |
| `trend_big_blocked_without_1d_ma7_rebreak` | :6016 | HARD | 同上 |
| `four_h_trend_waiting_for_15m_selector` | :6383 | HARD | → watchable + 到位自动升级，不丢候选 |
| `short_blocked_4h_lower_band_chase` | :6114 | HARD | → 追空惩罚改为强制更近止损 + 0.5× |
| `zec_short_blocked_confirmed_4h_low_chase` | :6051 | HARD, **硬编码 symbol** | → 迁入 config symbol profile（见 §8 风险 R4），逻辑本身合理 |

### 2.3 维持 SOFT 的（已经是软的，确认不升级）

- `v5_loss_prevention.py:91–159` 的 9 个标签（mixed_structure_conflict、m15_guard_broken 等）——**代码确认为 report-only**，正确定位，保留。
- `candidates.py:212` 的 8 个质量过滤（sentiment_below_min、funding_negative 等）——已是评分型，保留。
- v6 的 8 个 `v5_wait_*` watchable 门槛（:5741–5885）——保留 watchable 语义，但需要确保「wait」候选进入观察队列而非静默丢弃。

### 2.4 Over-gating verdict / 过度门控裁定

**成立（高置信度，代码结构证据）**：39 个硬门槛在 `_check_entry_structure_block` 链中串行执行，独立通过率相乘。即使每关 80% 通过率，14 关串行 ≈ 4.4% 综合通过率——与 *[PDF-cited]* 0.51% 覆盖率同量级。且多线共振持有（`short_support_confluence_hold` :6669 / `long_resistance_confluence_hold` :6674）与二测确认（:6314/:6324）本质是「结构强 → 不准做」，把最好的 structural pullback 机会拦掉了，方向上正好和四模式入场哲学相反。

**但 force-entry 54.55% 胜率不可直接采信为上限** *[PDF-cited]*：无组合约束、无叠仓限制、未计滑点资金费。真实可得空间在两者之间，需 §4 的 A/B 定量。

---

## 3. Gann Integration Design / Gann 集成设计（绿地规范）

快照零 Gann 代码 → 以下为新模块规范，落点 `trading_system/gann_feature.py`。

### 3.1 GannFeature schema

```python
@dataclass
class GannFeature:
    symbol: str
    zone: str          # above_1x1 | 1x2_to_1x1 | below_1x2 | unknown
    bias: str          # support_long | support_short | neutral | conflict
    confidence: float  # 0-1: f(样本长度, anchor稳定性, 跨symbol一致性, OOS表现)
    anchor_high: float; anchor_low: float
    anchor_ts: str; cycle_days: int
    scale: str         # log | linear
    line_values: dict  # {"1x1": px, "1x2": px, "2x1": px}
    anchor_stability: float  # ±3天窗口平移后 1x1 线的最大漂移 %
    stale: bool
```

**只输出结构化状态，不输出订单。**

### 3.2 接入点（唯一）

仓位乘数，接在 v6 sizing 计算处（macro multiplier 同层），**不进入 §2 门槛链**：

- bias 顺向且 confidence ≥ 0.6 → ×1.10–1.25（随 confidence 线性）
- bias 逆向 → ×0.50–0.75
- unknown / stale / confidence < 0.4 → ×1.00
- **禁止** Gann 绕过 BB/MA/Fib/model/risk 任何一层（任务书红线）

### 3.3 Per-symbol enablement

| Symbol | 启用 | 说明 |
|---|---|---|
| BTC/ETH | ✅ 全功能 | 历史长、anchor 稳定，首批试点 |
| SOL/ZEC | ⚠️ 反弹感知 | below_1x2 不得机械解读为空头延续；ZEC below_1x2 反而常是反弹区 *[PDF-cited]* → below_1x2 时 bias 强制 neutral，除非 OOS 证据支持 |
| HYPE | ⚠️ confidence 上限 0.5 | 历史短 |

### 3.4 Anchor 稳定性检验（上线前置条件）

每个信号日，anchor 搜索窗口 ±3 天平移，测 1x1 线值漂移。漂移 > 2%（BTC/ETH）或 > 5%（山寨）→ 该 symbol 的 Gann 只作解释性输出（confidence=0），不参与 sizing。

### 3.5 既有 A/B 证据解读 *[PDF-cited]*

低权重加分不改变行为（弱分数穿不透硬门槛链——与 §2.4 结构分析一致）；仓位调节小幅改善且方向正确；强过滤 n=1 无统计意义。→ **结论支持 macro-only 定位**，与任务书预设一致。

---

## 4. A/B Test Plan / A/B 测试计划

### 4.1 五臂对照

| Arm | 配置 |
|---|---|
| A0 | V6 baseline（现行 39 硬门槛） |
| A1 | V6 + Gann 低权重评分 |
| A2 | V6 + Gann 仓位乘数（§3.2） |
| A3 | V6 + Gann 强过滤（预期淘汰，留作对照） |
| A4 | **V6 + 重分类门槛层级**（§2 矩阵）± Gann 乘数 |

### 4.2 方法学硬要求（修补现有 A/B 的缺口）

1. **Next-bar 入场**：信号 bar 收盘 → 下一根开盘成交，杜绝 lookahead
2. **组合级回放**：并发仓位上限、单 symbol 上限、总风险预算——force-entry 的胜率幻觉主要死在这里
3. **成本模型**：taker fee + 资金费 + 冲击滑点（按 symbol 流动性分档）
4. **OOS 切分**：训练/验证按时间切，验证段 ≥ 30% 且包含至少一个方向切换段
5. **跨 symbol 一致性**：BTC/ETH/SOL/ZEC/HYPE 分组报告，防单币 carry
6. **样本门槛**：任何臂 trades < 30 → 结论标记不可靠（A3 的教训）

### 4.3 指标（任务书全集）

return, max drawdown, PF, win rate, avg return, worst trade, trade count, candidate coverage —— 另加：**blocked-candidate 影子收益**（被拦候选按 next-bar 虚拟成交跟踪 MFE/MAE），这是量化「每个门槛拦掉多少钱」的关键，直接反哺 §2 矩阵迭代。

### 4.4 复盘闭环（Review Scope #5）

每条候选落库字段：timestamp, symbol, side, indicator_level, model_prob, gann_zone, smart_signal, market_flow, entry, MFE, MAE, exit_reason, blocked_reason（若拦）。现有 `v5_loss_prevention` 标签体系（:48-200）可直接复用为标签源。

---

## 5. Telegram Message Spec / Telegram 消息规范

### 5.1 现状与目标差距（代码证据）

任务书目标「V5/V6 bot 只播执行+错误」**尚未实现**：`telegram_monitor_v5_v6.py:5060-5177` 命令路由仍在 V5/V6 监控进程内响应 `p/fc/report/news`。去重/限流基础设施**已存在且可复用**：`should_send_alert()`(:5323)、`summary_dedupe_digest()`(:5248)、新闻去重(:4212)、静默错误过滤 `is_silent_execution_error()`(:595-611)。

### 5.2 模板

**执行（V5/V6 bot）**：
```
[Somnus执行] V6
💹 BTCUSDT 开多
入场=65846.76 止损=63871.35 止盈=71415.74
成交=0.0032 订单=123456 状态=✅
理由=4H收复MA20；15M回踩持稳；Gann宏观不冲突；资金流中性偏多
```

**错误（V5/V6 bot，去重后）**：
```
[Somnus错误] V6
❌ BTCUSDT 平空 失败
码=-1007 超时；状态=未知；对账=进行中
动作=交易所状态确认前不重发
```
要求：保留 symbol/action/error code/order id/reconcile state；同错误 1h 内不重复（`should_send_alert` 现成）；-1121/-1007 未对账前静默（现有 :595 逻辑保留）。

**Xuejian `report`**（沿用现有 `build_report_response()`:3272 骨架）：数据时间戳必须显示；缓存/过期必须标注「数据过期/等待刷新」；模型概率、4H、1D 冲突时**分层陈述冲突来源**，不给单一结论（AGENTS.md 输出规则已有此要求，模板层面固化）。

### 5.3 迁移

`p/fc/report/news` 路由从 `telegram_monitor_v5_v6.py` 摘除 → 仅保留在 `hermes_somnus_command.py`；V5/V6 监控收到命令时回一条静态提示（`command_help()`:1946 已是此文案，直接作为唯一响应）。

---

## 6. Staged Implementation Plan / 分阶段实施

| Phase | 内容 | 出口条件 |
|---|---|---|
| **P1 只记录** | ① 门槛链插桩：每个 gate 记录 (candidate, gate, pass/fail, would-be-size)；② blocked-candidate 影子收益落库；③ GannFeature 只算不用；④ 修复快照断链 import（若在生产库同样存在） | 2 周数据；每个 gate 有拦截率 + 影子 PnL |
| **P2 影子对比** | A4 臂（重分类门槛）与 A0 并行跑影子账本；Gann 乘数影子生效 | 影子 A4 相对 A0：coverage ≥5×，PF 不降，MaxDD 不升 >20% |
| **P3 小仓实盘** | A4 以 0.25× 正常仓位实盘，限 BTC/ETH；用户逐笔可见 blocked→allowed 的差异单 | ≥30 笔；win rate 与影子偏差 <10pp |
| **P4 默认切换** | A4 成为默认；旧门槛链保留 feature-flag 可回滚 | 用户明确批准 |

**每阶段红线**：不混 V5/V6 key/账本/bot；Binance 418/429 不加频；账实不符先对账后交易。

---

## 7. Risk Checklist / 风险清单

- [ ] **R1 lookahead**：所有回放 next-bar 成交？信号时间戳 < 成交时间戳逐条校验？
- [ ] **R2 幸存者偏差**：force-entry 对照含已下架/流动性枯竭 symbol？
- [ ] **R3 门槛降级的尾部风险**：macro_mismatch 降级后，用 2022-式单边行情段做压力回放（逆宏观 0.5× 仓在趋势日的连损分布）
- [ ] **R4 硬编码 symbol 规则**：ZEC(:6051)、HYPE(:6230) 规则迁 config 后行为等价性测试
- [ ] **R5 快照/生产漂移**：本评审基于 56d1816；生产库行号与逻辑可能已漂移，实施前 diff 确认
- [ ] **R6 模型特征偏斜**：coin_scout_model.py:46-71 自述训练/推理特征分布不一致（LaNA AI 特征训练时缺失）——门槛降级会放大模型权重，此偏斜需先量化
- [ ] **R7 账本分离**：V5/V6 账本无自动同步——P3 小仓实盘期间人工对账节奏定义
- [ ] **R8 Telegram 迁移**：命令摘除后确认 Hermes 侧全部命令冒烟通过（AGENTS.md 测试命令节）
- [ ] **R9 密钥**：全程不落任何 key/token 到 git（本快照已验证无泄漏）

---

## 8. Additional Findings / 评审附带发现

1. **v6_execution.py 是 605KB 单体**，门槛、下单、风控、账本、图表混在一文件。P1 插桩前建议先把门槛链抽成 gate registry（每个 gate 一个对象：id、classification、predicate、metrics hook），否则 §2 矩阵无法程序化维护。
2. **快照 export 工具有 bug**：删模块没删 import（execution.py:15、telegram_monitor_v5_v6.py:37），下次导出建议加 `python -m compileall` 门禁。
3. **信号层边界干净**（好消息）：9 个信号源全部确认「只产证据、不触发订单」，无一绕过策略层——任务书关切的 Design Intent #3 现状已达标。
4. **`fib_structure_confirmation_missing`(:6495) 与 BB 共振门槛存在重复拦截**：同一「结构未确认」语义被 3 个 gate 重复表达，合并后矩阵可从 63 → ~50。
