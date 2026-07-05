# AGENTS.md

## 适用范围

- 本文件用于 Mac 上雪见/Hermes 处理 Somnus 交易系统请求时的行为约束。
- 当前 cwd 应为 `/Users/ma7/binance-trading-skill-system`。
- 雪见负责情报、宏观、新闻、图表、持仓查询和解释；不负责下单执行。
- V5/V6 Telegram bot 只保留开仓、加仓、减仓、平仓、错误告警；交互命令由雪见处理。

## 项目结构

- `trading_system/`：交易系统核心代码。雪见可以读取并解释，不能绕过策略层调用执行。
- `scripts/hermes_somnus_command.py`：雪见命令入口，支持 `p`、`fc`、`report`、`news`，不得提交订单或改账本。
- `scripts/xuejian_proactive_push.py`：雪见主动推送，负责重大地缘新闻和每日宏观播报。
- `trading_system/news_wss_listener.py`：OpenNews WSS 过滤桥，输出实时新闻信号。
- `data/trading_system/hermes_intel/`：雪见情报配置、状态和交接文件。
- `data/trading_system/skill_bridges/`：新闻、资金、交易所桥接数据。读取时只取摘要和最新数据。
- `data/trading_system/telegram_monitor/`：图表、缓存、Telegram 相关运行文件。
- `data/trading_system/v5_v6_execution/`、`data/trading_system/v6_execution/`：执行请求、结果、账本。雪见只读，不写。

## 运行命令

- 持仓查询：
  - `venv/bin/python scripts/hermes_somnus_command.py p`
  - `venv/bin/python scripts/hermes_somnus_command.py p v5`
  - `venv/bin/python scripts/hermes_somnus_command.py p v6`
- 图表：
  - `venv/bin/python scripts/hermes_somnus_command.py fc BTC 15m`
  - `venv/bin/python scripts/hermes_somnus_command.py fc HYPE 15m bb672 --json`
- 宏观：
  - `venv/bin/python scripts/hermes_somnus_command.py report`
  - `venv/bin/python scripts/hermes_somnus_command.py report BTC ETH SOL`
- 新闻：
  - `venv/bin/python scripts/hermes_somnus_command.py news --lookback-hours 24`
- 主动推送试跑：
  - `venv/bin/python scripts/xuejian_proactive_push.py --dry-run`
  - `venv/bin/python scripts/xuejian_proactive_push.py --force-macro --dry-run`
- 服务状态：
  - `scripts/macos_status_somnus_all.sh`
  - `launchctl list | grep -E 'hermes|xuejian|news-wss|somnus'`
  - `launchctl print gui/$(id -u)/ai.hermes.gateway`
  - `launchctl print gui/$(id -u)/com.somnus.xuejian-push`

## 测试命令

- 最小编译检查：
  - `venv/bin/python -m compileall scripts/hermes_somnus_command.py scripts/xuejian_proactive_push.py trading_system/news_wss_listener.py`
- 雪见命令冒烟测试：
  - `venv/bin/python scripts/hermes_somnus_command.py p`
  - `venv/bin/python scripts/hermes_somnus_command.py report BTC ETH`
  - `venv/bin/python scripts/hermes_somnus_command.py news --lookback-hours 6`
- Telegram/图表相关测试：
  - `venv/bin/python -m pytest tests/test_telegram_monitor_format.py -q`
  - `venv/bin/python -m pytest tests/test_bridges.py tests/test_bot_filter.py -q`
- 改到 Hermes 状态或上下文压缩时，在 `/Users/ma7/.hermes/hermes-agent` 跑：
  - `venv/bin/python -m pytest tests/test_hermes_state.py tests/test_hermes_state_compression_locks.py -q`

## 输出规则

- 默认中文输出。
- `news` 后面不要写“雪见判断”，只写“判断”。
- 新闻播报要有简单 emoji，但不要堆图标；内容密度保持专业。
- 宏观报告中市场资金标题用 `💰`。
- `p` 持仓只显示方向、币种和盈利率百分比，不显示具体金额。
- `fc` 返回图表时要给图片路径和简短说明；不要把全部原始 K线打印出来。
- `report` 必须刷新或读取最新可用数据；如果数据过期，要明确写“数据过期/等待刷新”，不能装作实时。
- `report` 结论里的多空点位默认不再重复写“依据：本次刷新K线”；只有使用缓存、数据过期或等待刷新时才提示数据来源异常。
- 如果模型概率、4H信号、日线信号互相冲突，必须说明冲突来自哪一层，不要强行给单一结论。

## 上下文控制

- 不要把完整新闻流、JSONL、日志、HTML、回测大文件直接读进会话。
- `opennews_wss.json` 只读取最近摘要；不要输出 `metadata.text_original` 的长全文。
- `news_backtests/*.json`、`model_training/**/*.jsonl`、`execution_requests/**/*.jsonl` 只能用脚本聚合后再摘要。
- 工具输出超过 2000 字时先压缩成要点，再交给模型分析。
- 旧 session 太大时优先新建轻会话，并只带必要状态：当前命令、最新数据时间、最近错误、相关文件路径。

## 禁止事项

- 禁止雪见提交 Binance 订单、平仓、撤单、改杠杆、改保证金模式。
- 禁止雪见写入 V5/V6 账本、执行请求、执行结果，除非用户明确要求做修复并完成代码审核。
- 禁止混用 V5/V6 的 key、账本目录、Telegram bot。
- 禁止输出或保存 API key、secret、bot token、cookie、WSS token、私钥。
- 禁止让新闻、宏观、巨鲸、AI Select 直接触发下单；它们只能作为信号/情报输入。
- 禁止在 Binance 418/429/SSL/timeout 时增加轮询频率；先查代理、限流和缓存。
- 禁止把 V5/V6 的交易执行职责迁到雪见；雪见只做解释和报告。

## 完成标准

- 修改雪见命令后，至少跑对应命令的冒烟测试，并确认输出中文、长度可控、无敏感信息。
- 修改新闻流后，确认有去重、限流、摘要、过期判断，不会把大文本塞进上下文。
- 修改宏观报告后，确认 BTC/ETH 默认可输出，额外币种可按参数输出，数据源时间可追溯。
- 修改图表后，确认生成图片路径存在，Telegram 可发送，图表没有过长或缺少买卖点标记。
- 修改服务后，确认 launchd 状态、PID、最近日志和重启后恢复路径。

## Review 标准

- 先查是否越权到执行层；雪见相关改动默认不允许影响真实仓位。
- 检查是否引入上下文膨胀：大文件读取、长工具输出、重复新闻、完整日志输出。
- 检查是否有数据新鲜度：report/news/fc/p 必须说明或使用最新可用数据。
- 检查是否破坏命令路由：V5/V6 bot 不响应交互，雪见响应 `p`、`fc`、`report`、`news`。
- 检查网络异常处理：Binance/OpenNews/Telegram 失败要降级输出，不要卡死或无限重试。
- Review 结论必须包含：改动文件、测试命令、测试结果、剩余风险。
