from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from uuid import uuid4

from trading_system.binance import BinanceAccessError, BinanceFuturesClient
from trading_system.bridges import BridgeLoadResult, BinanceTopMoversBridge, CoinGeckoTrendingBridge, ManualSentimentBridge, OpenNewsBridge, OpenTwitterBridge, OptionsDeliveryBridge, SkillBridgeDirectory, SquareFeedBridge
from trading_system.candidates import build_candidate_basket
from trading_system.config import StrategyConfig
from trading_system.engine import TradingSystem
from trading_system.execution_bridge import BinanceSkillExecutionBridge, read_execution_results, summarize_execution_results
from trading_system.issues import IssueRegistry, build_default_issue_registry, sync_live_access_issue
from trading_system.meme_radar import build_meme_radar, write_daily_scan_archive
from trading_system.macro_trend import MacroTrendSignal, compute_macro_trend
from trading_system.oi_divergence_system import OIDivergencePaperSystem
from trading_system.parameter_optimizer import run_bayesian_parameter_search
from trading_system.regime import detect_market_regime
from trading_system.reports import write_markdown_report
from trading_system.self_train_predictor import train_predictive_model
from trading_system.factor_backtest import distill_factor_training_samples, run_factor_backtest
from trading_system.self_train_dataset import collect_closed_trade_samples, collect_decision_samples, append_new_samples, collect_distill_process_samples, build_analysis_maturity_summary
from trading_system.coin_scout_trainer import train_from_existing_data as coin_scout_train
from trading_system.coin_scout_runtime import CoinScoutRuntime, WhaleScoutRuntime
from trading_system.sentiment_aggregator import aggregate_sentiment
from trading_system.skill_adapters import SkillBridgeSync
from trading_system.square_analysis import enrich_square_feed
from trading_system.square_reader import build_reader_from_export, parse_request_export
from trading_system.persistence import atomic_write_json, load_state


logger = logging.getLogger(__name__)


CAPABILITY_REQUIREMENTS = {
    "dex_execution": {"binance": set(), "okx": set()},
    "market_data": {"binance": {"crypto-market-rank", "meme-rush"}, "okx": {"okx-dex-market", "okx-dex-token", "okx-dex-trenches"}},
    "signal": {"binance": {"trading-signal"}, "okx": {"okx-dex-signal"}},
    "ws_realtime": {"binance": set(), "okx": {"okx-dex-ws"}},
    "security": {"binance": set(), "okx": {"okx-security"}},
    "defi": {"binance": set(), "okx": {"okx-defi-invest", "okx-defi-portfolio"}},
    "portfolio": {"binance": set(), "okx": {"okx-wallet-portfolio"}},
    "execution_gateway": {"binance": set(), "okx": {"okx-onchain-gateway"}},
}


class TradingPipeline:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.issues: IssueRegistry = build_default_issue_registry()
        # 加载 shadow 模型（对所有 candidate 打分，不拦截）
        from trading_system.execution import (
            _SHADOW_MODEL_PATHS,
            _COIN_FEATURE_NAMES,
            _shadow_feature_row,
        )
        import hashlib, json as _json
        self._shadow_models = {}
        self._shadow_model_meta = {}
        for name, path in _SHADOW_MODEL_PATHS.items():
            if not path.exists():
                self._shadow_model_meta[name] = {"status": "not_found", "model_path": str(path)}
                continue
            try:
                import pickle
                with open(path, "rb") as f:
                    content = f.read()
                model = pickle.loads(content)
                if isinstance(model, dict):
                    model = model.get("model", model)
                self._shadow_models[name] = model
                stat = path.stat()
                md5 = hashlib.md5(content).hexdigest()[:12]
                meta = {
                    "status": "loaded",
                    "model_path": str(path),
                    "file_size_bytes": stat.st_size,
                    "file_mtime": stat.st_mtime,
                    "file_md5_prefix": md5,
                    "feature_schema_version": "v1_65dim",
                }
                summary_path = path.parent / "ablation_result.json"
                if summary_path.exists():
                    try:
                        s = _json.loads(summary_path.read_text())
                        meta["train_samples"] = s.get("train_samples", 0)
                        meta["val_report"] = {
                            "accuracy": s.get("val_report", {}).get("accuracy"),
                            "macro_f1": s.get("val_report", {}).get("macro avg", {}).get("f1-score"),
                        }
                    except Exception:
                        pass
                self._shadow_model_meta[name] = meta
                logger.info("Pipeline shadow model loaded: %s (md5=%s)", name, md5)
            except Exception as e:
                logger.warning("Failed to load pipeline shadow model %s: %s", name, e)
                self._shadow_model_meta[name] = {"status": "load_error", "model_path": str(path), "error": str(e)}
        self._shadow_feature_names = _COIN_FEATURE_NAMES

    def run(
        self,
        mode: str = "live",
        square_curl_file: Path | None = None,
        collect_square_feed: bool = False,
    ) -> dict:
        run_id = uuid4().hex
        logger.info("pipeline run started: run_id=%s mode=%s", run_id, mode)
        skill_sync_summary = SkillBridgeSync(self.config.skill_raw_dir, self.config.skill_bridge_dir).sync()
        if collect_square_feed and square_curl_file is not None:
            self._collect_square_feed(square_curl_file)
        if self.config.square_feed_report_file.exists():
            enrich_square_feed(self.config.square_feed_report_file, self.config.square_enriched_report_file)

        options_delivery_result = (
            OptionsDeliveryBridge(
                underlyings=self.config.options_delivery_underlyings,
                limit=self.config.options_delivery_limit,
                snapshot_path=self.config.options_delivery_report_file,
            ).load()
            if self.config.options_delivery_enabled
            else BridgeLoadResult("options_delivery", [], {"status": "disabled_by_config", "count": 0})
        )
        base_bridge_results = [
            ManualSentimentBridge().load(self.config.sentiment_file),
            SquareFeedBridge(
                max_age_minutes=self.config.square_feed_max_age_minutes,
                stale_hotness_multiplier=self.config.square_feed_stale_hotness_multiplier,
            ).load(self.config.square_enriched_report_file),
            CoinGeckoTrendingBridge().load(),
            BinanceTopMoversBridge(self.config.base_url).load(),
            options_delivery_result,
            SkillBridgeDirectory().load(self.config.skill_bridge_dir),
        ]
        aggregation = aggregate_sentiment(base_bridge_results, self.config.merged_sentiment_file)
        runtime_config = replace(self.config, sentiment_file=self.config.merged_sentiment_file)
        if runtime_config.macro_trend_enabled:
            macro_trend = compute_macro_trend(
                BinanceFuturesClient(runtime_config.base_url),
                cache_file=runtime_config.macro_trend_cache_file,
                cache_ttl_seconds=runtime_config.macro_trend_cache_ttl_seconds,
                ma_period=runtime_config.macro_trend_ma_period,
                ma20_period=runtime_config.macro_trend_ma20_period,
                neutral_band_pct=runtime_config.macro_trend_neutral_band_pct,
                slope_neutral_threshold_pct=runtime_config.macro_trend_slope_neutral_threshold_pct,
            ).to_dict()
        else:
            macro_trend = MacroTrendSignal.neutral(source="disabled").to_dict()
        analysis_mode = mode
        execution_mode = analysis_mode
        system = TradingSystem(config=runtime_config, mode=analysis_mode)
        try:
            ranked = system.scan()
            if analysis_mode == "live":
                sync_live_access_issue(self.issues)
        except BinanceAccessError as exc:
            if analysis_mode == "live":
                sync_live_access_issue(self.issues, exc)
            raise

        candidate_symbols = [item.symbol for item in ranked[: self.config.candidate_confirm_limit] if item.score > 0]

        targeted_bridge_results = []
        if candidate_symbols and self.config.disable_open_token_bridges:
            targeted_bridge_results = [
                BridgeLoadResult(
                    "opennews",
                    [],
                    {
                        "status": "disabled_by_config",
                        "reason": "TS_DISABLE_OPEN_TOKEN_BRIDGES=1",
                        "target_symbols": candidate_symbols,
                    },
                ),
                BridgeLoadResult(
                    "opentwitter",
                    [],
                    {
                        "status": "disabled_by_config",
                        "reason": "TS_DISABLE_OPEN_TOKEN_BRIDGES=1",
                        "target_symbols": candidate_symbols,
                    },
                ),
            ]
        elif candidate_symbols:
            targeted_bridge_results = [
                OpenNewsBridge(
                    token=self.config.opennews_tokens,
                    lookback_hours=self.config.opennews_lookback_hours,
                    page_limit=self.config.targeted_opennews_page_limit,
                    max_pages=self.config.targeted_opennews_max_pages,
                    snapshot_path=self.config.opennews_report_file,
                ).fetch_for_symbols(candidate_symbols),
                OpenTwitterBridge(
                    token=self.config.twitter_tokens,
                    max_results=self.config.targeted_twitter_max_results,
                    snapshot_path=self.config.opentwitter_report_file,
                ).fetch_for_symbols(candidate_symbols),
            ]
            aggregation = aggregate_sentiment(base_bridge_results + targeted_bridge_results, self.config.merged_sentiment_file)
            runtime_config = replace(self.config, sentiment_file=self.config.merged_sentiment_file)
            system = TradingSystem(config=runtime_config, mode=analysis_mode)
            try:
                ranked = system.scan()
                if analysis_mode == "live":
                    sync_live_access_issue(self.issues)
            except BinanceAccessError as exc:
                if analysis_mode == "live":
                    sync_live_access_issue(self.issues, exc)
                raise
            candidate_symbols = [item.symbol for item in ranked[: self.config.candidate_confirm_limit] if item.score > 0]

        candidate_basket = build_candidate_basket(ranked)
        hotness_priority = self._build_hotness_priority(aggregation.signals, ranked)
        meme_radar = build_meme_radar(
            ranked,
            candidate_basket,
            hotness_priority,
            self.config.square_enriched_report_file,
        )
        meme_radar_archive_file = write_daily_scan_archive(
            meme_radar,
            self.config.meme_radar_archive_dir,
        )
        oi_divergence_candidates = self._build_oi_divergence_candidates(ranked)
        oi_divergence_paper = OIDivergencePaperSystem(runtime_config).rebalance(ranked)
        alerts = self._build_anomaly_alerts(ranked)
        market_regime = detect_market_regime(ranked, runtime_config)
        source_independence_summary = self._build_source_independence_summary(aggregation.signals)
        factor_backtest_summary = run_factor_backtest(
            runtime_config.closed_trade_dataset_file,
            runtime_config.factor_backtest_file,
        )
        factor_distill_summary = distill_factor_training_samples(
            runtime_config.closed_trade_dataset_file,
            factor_backtest_summary,
            runtime_config.factor_distilled_training_file,
        )
        # The predictive model must train on the full closed-trade ledger.  The
        # factor-distilled file is useful as a diagnostic/noise report, but using
        # it as the primary training set can freeze the model on a tiny filtered
        # subset and make the reported win rate look artificially stable.
        model_closed_dataset_file = runtime_config.closed_trade_dataset_file
        predictive_model_summary = train_predictive_model(
            runtime_config.decision_dataset_file,
            model_closed_dataset_file,
            runtime_config.predictive_model_file,
        )
        predictive_model_summary["training_dataset_file"] = str(model_closed_dataset_file)
        predictive_model_summary["factor_distill"] = factor_distill_summary
        if predictive_model_summary.get("status") == "ok":
            predictive_model_summary["factor_backtest"] = factor_backtest_summary
        bayes_opt_summary = run_bayesian_parameter_search(
            runtime_config.closed_trade_dataset_file,
            iterations=8,
            output_path=runtime_config.bayes_opt_output_file,
        )

        # --- Coin Scout 妖币模型训练 ---
        coin_scout_model_dir = Path("data/trading_system/model_training/coin_scout")
        coin_scout_summary = self._load_existing_coin_scout_summary(coin_scout_model_dir) or {"status": "skipped", "reason": "default"}
        if coin_scout_summary.get("status") == "skipped":
            try:
                coin_scout_summary = coin_scout_train(
                    closed_trades_path=runtime_config.closed_trade_dataset_file,
                    model_dir=coin_scout_model_dir,
                )
                coin_scout_summary["training_mode"] = "closed_trade_pnl_fallback"
                coin_scout_summary["retrained"] = True
            except Exception as exc:
                coin_scout_summary = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
                logger.warning("CoinScout training failed: %s", exc)

        # --- Whale overview 生产者: 为候选币抓真实聪明钱数据写每币缓存 ---
        # collection-only: 仅写 data_dir/whale_overview/{symbol}.json 供只读评估器用,
        # 不进入 rebalance/execution 真实交易决策。失败降级为文件缺失(消费者 fallback 轻量特征)。
        if getattr(self.config, "whale_overview_enabled", True) and hasattr(self.config, "data_dir"):
            try:
                from datetime import datetime, timezone
                from trading_system.whale_overview import WhaleOverviewConfig, build_whale_overview
                whale_dir = self.config.data_dir / "whale_overview"
                whale_dir.mkdir(parents=True, exist_ok=True)
                _ledger_path = whale_dir / "snapshot_ledger.jsonl"
                _ledger_ts = datetime.now(timezone.utc).isoformat()
                _produced = 0
                # V6 Phase2: 采集符号集 = 动态候选 ∪ 13主流币篮(去重)。
                # 主流币常被 universe 动量过滤(min_price_change_pct)挡在 ranked 外, 故强制每轮采集,
                # 向前累积精英聪明钱持仓到账本, 为 V6 主流币方向器备料。
                # collection-only: 只写 whale_overview/{symbol}.json + 账本, 不进 rebalance/execution。
                _majors_basket = [
                    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
                    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "TONUSDT", "UNIUSDT", "LTCUSDT",
                ]
                _candidate_syms = [s.symbol for s in ranked[: self.config.candidate_confirm_limit]]
                _collect_syms = list(dict.fromkeys(_candidate_syms + _majors_basket))
                for _sym in _collect_syms:
                    try:
                        _out_json = whale_dir / f"{_sym}.json"
                        build_whale_overview(
                            WhaleOverviewConfig(
                                symbol=_sym,
                                skill_bridge_dir=self.config.skill_bridge_dir,
                                output_json=_out_json,
                                output_html=whale_dir / f"{_sym}.html",
                            )
                        )
                        _produced += 1
                        # append-only 时序账本: 每币每轮快照留档, 不覆盖, 供 V6 蒸馏吃历史信号
                        try:
                            _ov = json.loads(_out_json.read_text())
                            with _ledger_path.open("a", encoding="utf-8") as _lf:
                                _lf.write(json.dumps(
                                    {"captured_at": _ledger_ts, "symbol": _sym, "overview": _ov},
                                    ensure_ascii=False,
                                ) + "\n")
                        except Exception as _le:
                            logger.debug("snapshot_ledger append failed for %s: %s", _sym, _le)
                    except Exception as _e:
                        logger.debug("whale_overview build failed for %s: %s", _sym, _e)
                logger.info("whale_overview produced: %d/%d (candidates+majors)", _produced, len(_collect_syms))
            except Exception as _e:
                logger.warning("whale_overview producer phase failed: %s", _e)

        # --- Coin Scout 实时评分（为每个 candidate 生成妖币概率）---
        coin_scout_evaluations = []
        try:
            scout_runtime = CoinScoutRuntime(model_dir="data/trading_system/model_training/coin_scout")
            if scout_runtime.model.is_trained:
                for snapshot in ranked[: self.config.candidate_confirm_limit]:
                    try:
                        # 获取 whale overview（如果已有缓存则复用）
                        whale_overview = None
                        whale_cache = (
                            self.config.data_dir / "whale_overview" / f"{snapshot.symbol}.json"
                            if hasattr(self.config, "data_dir")
                            else None
                        )
                        if whale_cache and whale_cache.exists():
                            whale_overview = json.loads(whale_cache.read_text())

                        decision = scout_runtime.evaluate(
                            snapshot=snapshot,
                            whale_overview=whale_overview,
                        )
                        coin_scout_evaluations.append({
                            "symbol": decision.symbol,
                            "action": decision.action,
                            "confidence": decision.confidence,
                            "predicted_label": decision.predicted_label,
                            "position_scale": decision.position_scale,
                            "reasons": decision.reasons,
                            "key_signals": decision.key_signals,
                        })
                    except Exception as e:
                        logger.debug("CoinScout eval failed for %s: %s", snapshot.symbol, e)
        except Exception as e:
            logger.warning("CoinScout evaluation phase failed: %s", e)

        # --- Whale Scout 聪明钱/巨鲸模型评估 ---
        whale_scout_evaluations = []
        try:
            whale_scout_runtime = WhaleScoutRuntime(model_dir="data/trading_system/model_training/whale_scout")
            if whale_scout_runtime.model.is_trained:
                for snapshot in ranked[: self.config.candidate_confirm_limit]:
                    try:
                        whale_overview = None
                        whale_cache = (
                            self.config.data_dir / "whale_overview" / f"{snapshot.symbol}.json"
                            if hasattr(self.config, "data_dir")
                            else None
                        )
                        if whale_cache and whale_cache.exists():
                            whale_overview = json.loads(whale_cache.read_text())

                        decision = whale_scout_runtime.evaluate(
                            snapshot=snapshot,
                            whale_overview=whale_overview,
                        )
                        whale_scout_evaluations.append({
                            "symbol": decision.symbol,
                            "action": decision.action,
                            "confidence": decision.confidence,
                            "predicted_label": decision.predicted_label,
                            "position_scale": decision.position_scale,
                            "reasons": decision.reasons,
                            "key_signals": decision.key_signals,
                            "model_source": "whale_scout",
                        })
                    except Exception as e:
                        logger.debug("WhaleScout eval failed for %s: %s", snapshot.symbol, e)
                logger.info("WhaleScout evaluation complete: %d evaluations", len(whale_scout_evaluations))
            else:
                logger.info("WhaleScout model not trained — skipping evaluation")
        except Exception as e:
            logger.warning("WhaleScout evaluation phase failed: %s", e)

        # --- Shadow model candidate scoring（对所有 candidate 打分，不拦截）---
        shadow_candidate_scores = self._score_candidates_with_shadow_models(ranked)

        self._reconcile_shadow_mirror_status(runtime_config)

        from trading_system.hf_signal_cache import load_hf_symbol_quality_cache
        hf_quality_lookup = load_hf_symbol_quality_cache(
            runtime_config.hf_backtest_entries_file,
            runtime_config.hf_backtest_report_file,
            ttl_seconds=runtime_config.hf_cache_ttl_seconds,
        )

        execution_system = TradingSystem(config=runtime_config, mode=execution_mode)
        try:
            try:
                rebalance = execution_system.rebalance(regime=market_regime, predictive_model=predictive_model_summary, hf_quality_lookup=hf_quality_lookup)
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                rebalance = execution_system.rebalance()
            sync_live_access_issue(self.issues)
        except BinanceAccessError as exc:
            sync_live_access_issue(self.issues, exc)
            raise

        # --- Training sample collection: append new closed trades & decision samples ---
        captured_at = None
        training_snapshot = {
            "macro_trend": macro_trend,
            "hotness_priority": hotness_priority,
            "oi_divergence_candidates": oi_divergence_candidates,
            "candidate_basket": candidate_basket,
            "meme_radar": meme_radar,
            "meme_radar_archive_file": str(meme_radar_archive_file),
            "top_ranked_symbols": [item.to_dict() for item in ranked[:15]],
            "paper_rebalance": rebalance,
        }
        try:
            from datetime import datetime, timezone
            captured_at = datetime.now(timezone.utc).isoformat()
            paper_state = load_state(runtime_config.state_file) if runtime_config.state_file.exists() else {}
            closed_samples, _ctx = collect_closed_trade_samples(paper_state, training_snapshot, captured_at=captured_at, hf_quality_lookup=hf_quality_lookup)
            if closed_samples:
                added_closed = append_new_samples(runtime_config.closed_trade_dataset_file, closed_samples)
            else:
                added_closed = 0
            decision_samples = collect_decision_samples(training_snapshot, captured_at=captured_at, hf_quality_lookup=hf_quality_lookup)
            if decision_samples:
                added_decisions = append_new_samples(runtime_config.decision_dataset_file, decision_samples)
            else:
                added_decisions = 0
            distill_samples = collect_distill_process_samples(paper_state, training_snapshot, captured_at=captured_at)
            if distill_samples:
                added_distill = append_new_samples(
                    Path(str(runtime_config.decision_dataset_file).replace("paper_decision_samples", "paper_distill_process_samples")),
                    distill_samples,
                )
            else:
                added_distill = 0
        except Exception as exc:
            added_closed = 0
            added_decisions = 0
            added_distill = 0
            self.issues.ensure(
                code="TRAINING_SAMPLE_COLLECTION_FAILED",
                severity="high",
                summary="Training sample collection failed.",
                detail=f"{type(exc).__name__}: {exc}",
                remediation="Inspect paper state/report schema and self_train_dataset collectors before trusting the next training report.",
            )
        sample_collection_summary = {
            "captured_at": captured_at,
            "closed_samples_added": added_closed,
            "decision_samples_added": added_decisions,
            "distill_process_samples_added": added_distill,
        }
        analysis_maturity_summary = build_analysis_maturity_summary(
            runtime_config.decision_dataset_file,
            runtime_config.closed_trade_dataset_file,
            runtime_config.state_file,
        )

        execution_bridge_summary = BinanceSkillExecutionBridge(runtime_config).write_requests(
            rebalance,
            [item.to_dict() for item in ranked],
        )
        execution_results = read_execution_results(self.config.execution_result_dir)
        execution_results_summary = summarize_execution_results(execution_results)
        execution_learning_summary = self._build_execution_learning_summary()
        performance_dashboard = self._build_performance_dashboard()
        unified_coverage = self._build_unified_coverage(skill_sync_summary)
        unified_signal_summary = self._build_unified_signal_summary(aggregation.signals)
        self._refresh_issue_statuses(skill_sync_summary, execution_results_summary)
        report = {
            "run_id": run_id,
            "mode": analysis_mode,
            "analysis_mode": analysis_mode,
            "execution_mode": execution_mode,
            "execution_realtime_enforced": execution_mode == "live",
            "square_feed_used": self.config.square_enriched_report_file.exists(),
            "merged_sentiment_file": str(self.config.merged_sentiment_file),
            "square_enriched_report_file": str(self.config.square_enriched_report_file),
            "bridge_summaries": aggregation.bridge_summaries,
            "skill_sync_summary": skill_sync_summary,
            "unified_coverage": unified_coverage,
            "unified_signal_summary": unified_signal_summary,
            "market_regime": market_regime,
            "macro_trend": macro_trend,
            "source_independence_summary": source_independence_summary,
            "predictive_model_summary": {k: v for k, v in predictive_model_summary.items() if not k.startswith("_")},
            "coin_scout_summary": coin_scout_summary,
            "coin_scout_evaluations": coin_scout_evaluations,
            "whale_scout_evaluations": whale_scout_evaluations,
            "shadow_candidate_scores": shadow_candidate_scores,
            "factor_backtest_summary": factor_backtest_summary,
            "factor_distill_summary": factor_distill_summary,
            "bayes_opt_summary": bayes_opt_summary,
            "opennews_report_file": str(self.config.opennews_report_file),
            "opentwitter_report_file": str(self.config.opentwitter_report_file),
            "options_delivery_report_file": str(self.config.options_delivery_report_file),
            "candidate_confirmation": {
                "symbols": candidate_symbols,
                "opennews_targeted": next((item.metadata for item in targeted_bridge_results if item.bridge_name == "opennews"), {}),
                "twitter_targeted": next((item.metadata for item in targeted_bridge_results if item.bridge_name == "opentwitter"), {}),
            },
            "hotness_priority": hotness_priority,
            "meme_radar": meme_radar,
            "meme_radar_archive_file": str(meme_radar_archive_file),
            "oi_divergence_candidates": oi_divergence_candidates,
            "oi_divergence_paper": oi_divergence_paper,
            "candidate_basket": candidate_basket,
            "alerts": alerts,
            "top_ranked_symbols": [item.to_dict() for item in ranked[:15]],
            "paper_rebalance": rebalance,
            "execution_bridge_summary": execution_bridge_summary,
            "sample_collection_summary": sample_collection_summary,
            "analysis_maturity_summary": analysis_maturity_summary,
            "execution_result_count": len(execution_results),
            "execution_results_summary": execution_results_summary,
            "execution_learning_summary": execution_learning_summary,
            "performance_dashboard": performance_dashboard,
            "issue_count": len(self.issues.items()),
            "issues": [item.to_dict() for item in self.issues.items()],
            "issues_file_json": str(self.config.issues_json_file),
            "issues_file_md": str(self.config.issues_md_file),
        }
        self._write_outputs(report)
        logger.info("pipeline run finished: run_id=%s issues=%s", run_id, report["issue_count"])
        return report

    def collect_square_feed_only(self, square_curl_file: Path) -> dict:
        self._collect_square_feed(square_curl_file)
        report = {"square_feed_report_file": str(self.config.square_feed_report_file)}
        self.issues.write(self.config.issues_json_file, self.config.issues_md_file)
        return report

    def collect_options_delivery_only(self) -> dict:
        result = OptionsDeliveryBridge(
            underlyings=self.config.options_delivery_underlyings,
            limit=self.config.options_delivery_limit,
            snapshot_path=self.config.options_delivery_report_file,
        ).load()
        report = {"options_delivery_report_file": str(self.config.options_delivery_report_file), **result.metadata}
        self.issues.write(self.config.issues_json_file, self.config.issues_md_file)
        return report

    def _score_candidates_with_shadow_models(self, ranked: list) -> dict:
        """
        对所有 candidate 用 shadow 模型打分。
        优先用 CoinScoutRuntime 构建完整特征（含 whale_overview），
        找不到 whale_overview 时 fallback 到轻量版 _snapshot_to_coin_features()。
        返回 {
          "scores": {symbol: {model_name: {label, prob_0, prob_1, prob_2, prob_moon, feature_source}}},
          "errors": {symbol: {model_name: error_msg}},
          "model_meta": {model_name: {md5_prefix, ...}},
        }
        """
        import json as _json
        import numpy as np
        from dataclasses import asdict as _asdict
        from trading_system.execution import PaperExecutionEngine, _shadow_feature_row
        from trading_system.coin_scout_runtime import CoinScoutRuntime

        result = {"scores": {}, "errors": {}, "model_meta": {}}
        # 写入模型元数据
        for name, meta in self._shadow_model_meta.items():
            m = {k: v for k, v in meta.items() if k != "val_report"}
            if "val_report" in meta:
                m["val_accuracy"] = meta["val_report"].get("accuracy")
                m["val_macro_f1"] = meta["val_report"].get("macro_f1")
            result["model_meta"][name] = m
        if not self._shadow_models:
            return result

        # 初始化 CoinScoutRuntime 用于构建完整特征
        runtime = CoinScoutRuntime(model_dir="data/trading_system/model_training/coin_scout")

        for snapshot in ranked[: self.config.candidate_confirm_limit]:
            try:
                # 优先：用 CoinScoutRuntime 构建完整特征
                whale_overview = None
                whale_cache = (
                    self.config.data_dir / "whale_overview" / f"{snapshot.symbol}.json"
                    if hasattr(self.config, "data_dir")
                    else None
                )
                if whale_cache and whale_cache.exists():
                    try:
                        whale_overview = _json.loads(whale_cache.read_text())
                    except Exception:
                        pass

                coin_features = None
                feature_source = "lightweight_fallback"
                if whale_overview is not None:
                    try:
                        coin_features = runtime.build_features_from_snapshot(
                            snapshot, whale_overview
                        )
                        feature_source = "full_runtime"
                    except Exception as e:
                        logger.debug(
                            "Full feature build failed for %s: %s, fallback to lightweight",
                            snapshot.symbol, e,
                        )

                # fallback：轻量版
                if coin_features is None:
                    coin_features = PaperExecutionEngine._snapshot_to_coin_features(snapshot)
                    feature_source = "lightweight_fallback"

                if coin_features is None:
                    continue

                cf_dict = _asdict(coin_features)
                symbol_scores = {}
                symbol_errors = {}
                for name, model in self._shadow_models.items():
                    try:
                        X = np.array([_shadow_feature_row(cf_dict, model)])
                        pred = int(model.predict(X)[0])
                        probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0, 0, 0]
                        prob_0 = float(probs[0]) if len(probs) > 0 else 0.0
                        prob_1 = float(probs[1]) if len(probs) > 1 else 0.0
                        prob_2 = float(probs[2]) if len(probs) > 2 else 0.0
                        symbol_scores[name] = {
                            "model_label": pred,
                            "prob_0": round(prob_0, 4),
                            "prob_1": round(prob_1, 4),
                            "prob_2": round(prob_2, 4),
                            "prob_moon": round(prob_1 + prob_2, 4),
                            "feature_source": feature_source,
                        }
                    except Exception as e:
                        symbol_errors[name] = f"{type(e).__name__}: {e}"
                        logger.debug("Shadow model %s failed for %s: %s", name, snapshot.symbol, e)
                if symbol_scores:
                    result["scores"][snapshot.symbol] = symbol_scores
                if symbol_errors:
                    result["errors"][snapshot.symbol] = symbol_errors
            except Exception as e:
                logger.debug("Shadow scoring failed for %s: %s", snapshot.symbol, e)
        return result

    def _reconcile_shadow_mirror_status(self, config: StrategyConfig) -> None:
        if not config.state_file.exists():
            return
        state = load_state(config.state_file)
        shadow_positions = [dict(item) for item in state.get("shadow_positions", []) if isinstance(item, dict)]
        if not shadow_positions:
            return
        results = read_execution_results(config.execution_result_dir)
        if not results:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        open_results_by_key: dict[str, dict] = {}
        open_results_by_symbol: dict[str, list[dict]] = {}
        for item in results:
            payload = item.get("payload", {})
            for result in payload.get("results", []):
                symbol = str(result.get("symbol") or "")
                meta = dict(result.get("meta") or {})
                action = str(meta.get("action") or "")
                status = str(result.get("status") or "")
                request_id = str(result.get("id") or "")
                error = str(result.get("error") or "")
                if action in {"open_long", "open_short"}:
                    record = {
                        "status": status,
                        "request_id": request_id,
                        "error": error,
                        "result_time": self._parse_mirror_result_time(result, payload),
                    }
                    shadow_position_key = str(meta.get("shadow_position_key") or "")
                    if shadow_position_key:
                        open_results_by_key[shadow_position_key] = record
                    if symbol:
                        open_results_by_symbol.setdefault(symbol, []).append(record)
        for items in open_results_by_symbol.values():
            items.sort(key=lambda item: item.get("result_time") or datetime.min.replace(tzinfo=timezone.utc))
        updated = False
        for position in shadow_positions:
            symbol = str(position.get("symbol") or "")
            current_mirror = str(position.get("binance_mirror_status") or "")
            if current_mirror in {"submitted", "submitted_unprotected"}:
                continue
            result = self._find_shadow_mirror_open_result(position, open_results_by_key, open_results_by_symbol)
            if not result:
                continue
            if result["status"] in {"submitted", "submitted_unprotected"}:
                position["binance_mirror_status"] = result["status"]
                position["binance_mirror_error"] = None
                position["binance_mirror_updated_at"] = now_iso
                position["binance_request_id"] = result["request_id"]
                updated = True
            elif result["status"] == "failed":
                position["binance_mirror_status"] = "submit_failed"
                position["binance_mirror_error"] = result["error"]
                position["binance_mirror_updated_at"] = now_iso
                position["binance_request_id"] = result["request_id"]
                updated = True
        if updated:
            state["shadow_positions"] = shadow_positions
            from trading_system.persistence import atomic_write_state
            atomic_write_state(config.state_file, state)
            logger.info("shadow mirror status reconciled: %d positions updated",
                        sum(1 for p in shadow_positions if p.get("binance_mirror_updated_at") == now_iso))

    def _find_shadow_mirror_open_result(
        self,
        position: dict,
        open_results_by_key: dict[str, dict],
        open_results_by_symbol: dict[str, list[dict]],
    ) -> dict | None:
        shadow_position_key = str(position.get("shadow_position_key") or "")
        if shadow_position_key and shadow_position_key in open_results_by_key:
            return open_results_by_key[shadow_position_key]

        symbol = str(position.get("symbol") or "")
        if not symbol:
            return None
        candidates = list(open_results_by_symbol.get(symbol, []))
        if not candidates:
            return None
        opened_at = self._parse_mirror_time(str(position.get("opened_at") or position.get("timestamp") or ""))
        if opened_at:
            lower_bound = opened_at - timedelta(minutes=5)
            upper_bound = opened_at + timedelta(minutes=30)
            candidates = [
                item for item in candidates
                if (
                    item.get("result_time") is not None
                    and lower_bound <= item["result_time"] <= upper_bound
                )
            ]
        if not candidates:
            return None
        return candidates[-1]

    @staticmethod
    def _parse_mirror_result_time(result: dict, payload: dict) -> datetime | None:
        for key in ("submitted_at", "generated_at"):
            parsed = TradingPipeline._parse_mirror_time(str(result.get(key) or ""))
            if parsed:
                return parsed
        return TradingPipeline._parse_mirror_time(str(payload.get("generated_at") or ""))

    @staticmethod
    def _parse_mirror_time(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _collect_square_feed(self, curl_file: Path) -> None:
        if not curl_file.exists():
            self.issues.ensure(
                code="SQUARE_CURL_FILE_MISSING",
                severity="medium",
                summary="Square curl export file is missing.",
                detail=f"Expected curl export at {curl_file}.",
                remediation="Export a fresh logged-in curl command from Binance Square and place it at the configured path.",
            )
            return
        try:
            _url, _headers, payload = parse_request_export(curl_file)
            reader = build_reader_from_export(curl_file)
            posts = reader.fetch_feed_posts(
                page_index=int(payload.get("pageIndex", 1)),
                page_size=int(payload.get("pageSize", 20)),
                scene=payload.get("scene", "web-homepage"),
                content_ids=payload.get("contentIds", []),
            )
            status = "ok"
            error = ""
        except Exception as exc:
            posts = []
            status = "error"
            error = str(exc)
            self.issues.ensure(
                code="SQUARE_FEED_REFRESH_FAILED",
                severity="medium",
                summary="Square feed refresh failed; Square-only signal is downweighted and cannot open alone.",
                detail=error,
                remediation="Refresh Binance Square browser session/curl export, then rerun collection.",
            )
        report = {
            "summary": {
                "post_count": len(posts),
                "source": "curl_export",
                "status": status,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
            },
            "posts": [post.to_dict() for post in posts],
        }
        self.config.square_feed_report_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.config.square_feed_report_file, report)

    def _write_outputs(self, report: dict) -> None:
        self.config.pipeline_report_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.config.pipeline_report_file, report)
        self.issues.write(self.config.issues_json_file, self.config.issues_md_file)
        write_markdown_report(report, self.config.pipeline_report_md_file)

    def _build_anomaly_alerts(self, ranked: list) -> list[dict]:
        alerts: list[dict] = []
        for snapshot in ranked:
            if snapshot.price_change_pct_5m < self.config.anomaly_price_change_pct_5m:
                continue
            flags: list[str] = []
            if snapshot.oi_change_pct_4h >= self.config.anomaly_oi_change_pct_4h:
                flags.append("oi_spike")
            if snapshot.quote_volume_usdt_24h >= self.config.anomaly_quote_volume_usdt_24h:
                flags.append("volume_spike")
            if (
                snapshot.funding_rate <= self.config.anomaly_negative_funding_rate
                or snapshot.funding_rate >= self.config.anomaly_positive_funding_rate
            ):
                flags.append("funding_spike")
            if not flags:
                continue
            alerts.append(
                {
                    "symbol": snapshot.symbol,
                    "severity": "high",
                    "price_change_pct_5m": round(snapshot.price_change_pct_5m, 4),
                    "price_change_pct_24h": round(snapshot.price_change_pct_24h, 4),
                    "oi_change_pct_4h": round(snapshot.oi_change_pct_4h, 4),
                    "quote_volume_usdt_24h": round(snapshot.quote_volume_usdt_24h, 4),
                    "funding_rate": round(snapshot.funding_rate, 8),
                    "trigger_flags": flags,
                    "summary": f"{snapshot.symbol} 5m涨幅{snapshot.price_change_pct_5m:.1f}% 且出现{','.join(flags)}",
                }
            )
        return alerts

    def _build_hotness_priority(self, signals: list, ranked: list) -> list[dict]:
        ranked_map = {item.symbol: item for item in ranked}
        hotness_rows: list[dict] = []
        allowed_sources = {"square_feed", "coingecko_trending", "binance_top_movers"}
        for signal in sorted(signals, key=lambda item: (item.hotness_score, item.mention_count), reverse=True):
            source_names = {part.strip() for part in str(signal.source).split(",") if part.strip()}
            if not (source_names & allowed_sources):
                continue
            snapshot = ranked_map.get(signal.symbol)
            priority = signal.hotness_score
            if snapshot is not None:
                priority += min(20.0, max(0.0, snapshot.price_change_pct_24h) * 0.4)
                priority += min(18.0, max(0.0, snapshot.oi_change_pct_4h) * 0.6)
            hotness_rows.append(
                {
                    "symbol": signal.symbol,
                    "priority_score": round(min(100.0, priority), 2),
                    "hotness_score": round(signal.hotness_score, 2),
                    "mentions": signal.mention_count,
                    "source": signal.source,
                    "notes": signal.notes,
                }
            )
        hotness_rows.sort(key=lambda item: (item["priority_score"], item["mentions"]), reverse=True)
        return hotness_rows[:15]

    def _build_oi_divergence_candidates(self, ranked: list) -> list[dict]:
        candidates: list[dict] = []
        for snapshot in ranked:
            if snapshot.setup_type != "oi_divergence_long_candidate":
                continue
            timing_label = "逼近启动点" if snapshot.oi_price_divergence_score >= 85 else "继续观察"
            candidates.append(
                {
                    "symbol": snapshot.symbol,
                    "oi_price_divergence_score": round(snapshot.oi_price_divergence_score, 2),
                    "oi_change_pct_4h": round(snapshot.oi_change_pct_4h, 3),
                    "price_change_pct_4h": round(snapshot.price_change_pct_4h, 3),
                    "sentiment_score": round(snapshot.sentiment_score, 2),
                    "score": round(snapshot.score, 2),
                    "timing_label": timing_label,
                }
            )
        candidates.sort(key=lambda item: (item["oi_price_divergence_score"], item["score"]), reverse=True)
        return candidates[:10]

    def _build_execution_learning_summary(self) -> dict:
        if not self.config.state_file.exists():
            return {"status": "missing", "state_file": str(self.config.state_file), "closed_trades": 0, "by_reason": []}
        payload = load_state(self.config.state_file)
        journal = payload.get("journal", [])
        closed = [item for item in journal if str(item.get("action")) in {"exit", "take_profit_partial"}]
        if not closed:
            return {"status": "ok", "state_file": str(self.config.state_file), "closed_trades": 0, "by_reason": []}
        buckets: dict[str, dict[str, float]] = {}
        for item in closed:
            reason = str(item.get("reason") or "unknown")
            key = reason.split(",", 1)[0].split(":", 1)[0]
            slot = buckets.setdefault(key, {"count": 0, "win_count": 0, "pnl_total": 0.0})
            pnl = float(item.get("pnl_pct") or 0.0)
            slot["count"] += 1
            slot["pnl_total"] += pnl
            if pnl > 0:
                slot["win_count"] += 1
        by_reason = []
        for reason, stats in sorted(buckets.items(), key=lambda item: (item[1]["pnl_total"], item[1]["count"]), reverse=True):
            count = int(stats["count"])
            pnl_total = float(stats["pnl_total"])
            by_reason.append(
                {
                    "reason": reason,
                    "count": count,
                    "win_rate": round((float(stats["win_count"]) / count) * 100.0, 2) if count else 0.0,
                    "avg_pnl_pct": round((pnl_total / count) * 100.0, 3) if count else 0.0,
                    "total_pnl_pct": round(pnl_total * 100.0, 3),
                }
            )
        return {
            "status": "ok",
            "state_file": str(self.config.state_file),
            "closed_trades": len(closed),
            "by_reason": by_reason[:10],
        }

    def _build_performance_dashboard(self) -> dict:
        """Stone/Nicjiang-style PnL board: low win-rate is acceptable only with positive expectancy."""
        if not self.config.state_file.exists():
            return {"status": "missing", "state_file": str(self.config.state_file)}
        payload = load_state(self.config.state_file)
        journal = payload.get("journal", [])
        positions = payload.get("positions", [])
        closed = [item for item in journal if str(item.get("action")) in {"exit", "take_profit_partial"}]
        realized_pnl_usdt = 0.0
        wins = 0
        losses = 0
        for item in closed:
            pnl_pct = float(item.get("pnl_pct") or 0.0)
            size_fraction = float(item.get("size_fraction") or 1.0)
            notional = float(item.get("notional_usdt") or 0.0)
            realized_pnl_usdt += notional * pnl_pct
            if pnl_pct > 0:
                wins += 1
            else:
                losses += 1
        unrealized_pnl_usdt = 0.0
        open_notional_usdt = 0.0
        for item in positions:
            entry = float(item.get("entry_price") or 0.0)
            current = float(item.get("current_price") or entry or 0.0)
            notional = float(item.get("notional_usdt") or 0.0)
            if entry <= 0 or notional <= 0:
                continue
            open_notional_usdt += notional
            unrealized_pnl_usdt += notional * ((current - entry) / entry)
        total_pnl_usdt = realized_pnl_usdt + unrealized_pnl_usdt
        closed_count = len(closed)
        win_rate = (wins / closed_count * 100.0) if closed_count else 0.0
        gross_basis = sum(float(item.get("notional_usdt") or 0.0) for item in closed)
        realized_return_pct = (realized_pnl_usdt / gross_basis * 100.0) if gross_basis else 0.0
        open_return_pct = (unrealized_pnl_usdt / open_notional_usdt * 100.0) if open_notional_usdt else 0.0
        total_basis = gross_basis + open_notional_usdt
        total_return_pct = (total_pnl_usdt / total_basis * 100.0) if total_basis else 0.0
        expectancy_pct = (sum(float(item.get("pnl_pct") or 0.0) for item in closed) / closed_count * 100.0) if closed_count else 0.0
        def _rolling_stats(window: int) -> dict:
            subset = closed[-window:]
            count = len(subset)
            wins_in_window = sum(1 for item in subset if float(item.get("pnl_pct") or 0.0) > 0)
            pnl_sum = sum(float(item.get("pnl_pct") or 0.0) for item in subset)
            return {
                "window": window,
                "closed_trades": count,
                "wins": wins_in_window,
                "losses": count - wins_in_window,
                "win_rate": round((wins_in_window / count * 100.0), 2) if count else 0.0,
                "expectancy_pct_per_closed_trade": round((pnl_sum / count * 100.0), 3) if count else 0.0,
            }
        rolling_10 = _rolling_stats(10)
        rolling_20 = _rolling_stats(20)
        if closed_count >= 8 and win_rate < 40.0 and total_pnl_usdt > 0 and expectancy_pct > 0:
            diagnosis = "低胜率高收益：允许，核心是让盈利单覆盖多笔小亏，继续强化分批止盈/奔跑仓。"
        elif closed_count >= 8 and win_rate < 40.0 and total_pnl_usdt <= 0:
            diagnosis = "低胜率低收益：不可接受，应收紧入场与止损，减少 churn 样本。"
        elif total_pnl_usdt > 0:
            diagnosis = "正期望：维持，按原因桶继续放大盈利路径。"
        else:
            diagnosis = "负期望：降频，先修复止损与入场质量。"
        return {
            "status": "ok",
            "state_file": str(self.config.state_file),
            "open_positions": len(positions),
            "closed_trades": closed_count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "realized_pnl_usdt": round(realized_pnl_usdt, 4),
            "realized_return_pct": round(realized_return_pct, 3),
            "unrealized_pnl_usdt": round(unrealized_pnl_usdt, 4),
            "unrealized_return_pct": round(open_return_pct, 3),
            "total_pnl_usdt": round(total_pnl_usdt, 4),
            "total_return_pct": round(total_return_pct, 3),
            "expectancy_pct_per_closed_trade": round(expectancy_pct, 3),
            "rolling_10": rolling_10,
            "rolling_20": rolling_20,
            "primary_win_rate_note": "win_rate is lifetime cumulative; use rolling_10/rolling_20 to detect live drift.",
            "diagnosis": diagnosis,
        }

    def _build_unified_coverage(self, skill_sync_summary: list[dict]) -> dict:
        available_names = {str(item.get("bridge_name") or "") for item in skill_sync_summary}
        coverage: dict[str, dict] = {}
        for category, requirements in CAPABILITY_REQUIREMENTS.items():
            binance_sources = sorted(requirements.get("binance", set()) & available_names)
            okx_sources = sorted(requirements.get("okx", set()) & available_names)
            missing_okx = sorted(requirements.get("okx", set()) - available_names)
            coverage[category] = {
                "binance_sources": binance_sources,
                "okx_sources": okx_sources,
                "mode": "parallel" if binance_sources and okx_sources else ("okx_only" if okx_sources else "binance_only" if binance_sources else "missing"),
                "parallel_enabled": bool(binance_sources and okx_sources),
                "okx_gap_filled": bool(okx_sources),
                "missing_okx_sources": missing_okx,
            }
        return coverage

    def _build_unified_signal_summary(self, signals: list) -> dict:
        by_symbol: dict[str, dict] = {}
        okx_markers = ("okx-",)
        binance_markers = ("square_feed", "manual_", "crypto-market-rank", "meme-rush", "trading-signal", "lanaaielsa-distill", "lanaai-square-distill", "gmgn-wallet-distill")
        for signal in sorted(signals, key=lambda item: item.hotness_score, reverse=True):
            source_names = [part.strip() for part in str(signal.source).split(",") if part.strip()]
            if not source_names:
                source_names = [str(signal.source).strip()]
            slot = by_symbol.setdefault(
                signal.symbol,
                {
                    "symbol": signal.symbol,
                    "score": signal.hotness_score,
                    "mentions": signal.mention_count,
                    "sources": [],
                    "parallel_sources": [],
                    "binance_count": 0,
                    "okx_count": 0,
                },
            )
            slot["score"] = max(slot["score"], signal.hotness_score)
            slot["mentions"] += signal.mention_count
            for source_name in source_names:
                if source_name not in slot["sources"]:
                    slot["sources"].append(source_name)
                if source_name.startswith(okx_markers):
                    slot["okx_count"] += 1
                if source_name.startswith(binance_markers) or source_name == "square_feed":
                    slot["binance_count"] += 1
        ranked = []
        for slot in by_symbol.values():
            if slot["binance_count"] and slot["okx_count"]:
                slot["parallel_sources"] = ["binance", "okx"]
            elif slot["okx_count"]:
                slot["parallel_sources"] = ["okx"]
            else:
                slot["parallel_sources"] = ["binance"]
            ranked.append(slot)
        ranked.sort(key=lambda item: (len(item["parallel_sources"]) == 2, item["score"], item["mentions"]), reverse=True)
        return {
            "top_symbols": ranked[:10],
            "parallel_symbol_count": sum(1 for item in ranked if len(item["parallel_sources"]) == 2),
            "okx_supported_symbol_count": sum(1 for item in ranked if "okx" in item["parallel_sources"]),
        }

    def _build_source_independence_summary(self, signals: list) -> dict:
        weak_sources = {"binance_top_movers", "coingecko_trending"}
        strong_sources = {"square_feed", "manual_distill", "manual_sentiment", "trading-signal", "crypto-market-rank", "meme-rush", "lanaaielsa-distill", "lanaai-square-distill", "gmgn-wallet-distill"}
        rows: list[dict] = []
        for signal in signals:
            signal_source = getattr(signal, "source", None)
            if signal_source is None and isinstance(signal, dict):
                signal_source = signal.get("source", "")
            signal_symbol = getattr(signal, "symbol", None)
            if signal_symbol is None and isinstance(signal, dict):
                signal_symbol = signal.get("symbol", "")
            signal_hotness = getattr(signal, "hotness_score", None)
            if signal_hotness is None and isinstance(signal, dict):
                signal_hotness = signal.get("hotness_score", 0.0)
            signal_mentions = getattr(signal, "mention_count", None)
            if signal_mentions is None and isinstance(signal, dict):
                signal_mentions = signal.get("mention_count", 0)
            source_names = [part.strip() for part in str(signal_source or "").split(",") if part.strip()]
            symbol = str(signal_symbol or "")
            score = float(signal_hotness or 0.0)
            mentions = int(float(signal_mentions or 0))
            if not source_names:
                continue
            strong = len([name for name in source_names if name in strong_sources])
            weak = len([name for name in source_names if name in weak_sources])
            independence_score = max(0.0, min(100.0, 25.0 + (strong * 28.0) - (weak * 18.0) + min(12.0, mentions * 0.6)))
            state = "independent" if strong > 0 else "price_loop_risk"
            if weak and not strong:
                state = "price_loop_risk"
            rows.append(
                {
                    "symbol": symbol,
                    "sources": source_names,
                    "independence_score": round(independence_score, 2),
                    "independence_state": state,
                    "hotness_score": round(score, 2),
                    "mentions": mentions,
                }
            )
        rows.sort(key=lambda item: (item["independence_state"] != "price_loop_risk", item["independence_score"]), reverse=False)
        return {
            "symbols": rows,
            "price_loop_risk_count": sum(1 for item in rows if item["independence_state"] == "price_loop_risk"),
        }

    @staticmethod
    def _load_existing_coin_scout_summary(model_dir: Path) -> dict | None:
        model_exists = (model_dir / "coin_scout_model.txt").exists() or (model_dir / "coin_scout_model.pkl").exists()
        if not model_exists:
            return {"status": "skipped", "reason": "coin_scout_model_missing", "retrained": False}
        for summary_name in ("training_summary_4h_v2.json", "training_summary_4h.json"):
            summary_path = model_dir / summary_name
            if not summary_path.exists():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(summary, dict) and summary.get("status") == "ok":
                summary["training_mode"] = "preserved_4h_klines"
                summary["summary_file"] = str(summary_path)
                summary["model_dir"] = str(model_dir)
                summary["retrained"] = False
                return summary
        return {"status": "skipped", "reason": "no_4h_coin_scout_summary", "retrained": False}

    def _refresh_issue_statuses(self, skill_sync_summary: list[dict], execution_results_summary: dict) -> None:
        available_bridges = {str(item.get("bridge_name") or "") for item in skill_sync_summary}
        expected_bridges = {"crypto-market-rank", "meme-rush", "trading-signal", "okx-dex-signal", "okx-dex-token", "okx-dex-market", "okx-dex-trenches", "okx-security", "okx-dex-ws", "okx-defi-invest", "okx-defi-portfolio", "okx-wallet-portfolio", "okx-onchain-gateway"}
        if expected_bridges.issubset(available_bridges):
            self.issues.remove("BINANCE_SKILLS_BRIDGES_PENDING")
            self.issues.remove("OKX_SKILLS_BRIDGES_PENDING")
        else:
            missing = sorted(expected_bridges - available_bridges)
            self.issues.ensure(
                code="OKX_SKILLS_BRIDGES_PENDING",
                severity="medium",
                summary="OKX bridge coverage is not complete yet.",
                detail=f"Missing bridge inputs: {', '.join(missing)}",
                remediation="Import the missing OKX skill outputs into data/trading_system/skill_raw so Binance and OKX signals can run in parallel.",
            )
        if execution_results_summary.get("actionable_result_count", 0) > 0:
            self.issues.remove("EXECUTION_PAPER_ONLY")
