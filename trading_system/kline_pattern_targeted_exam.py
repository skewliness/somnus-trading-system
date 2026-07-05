"""针对指定币种的形态识别训练+考试，带详细日志输出。

找每个币历史涨幅最高的时段，在该时段前后训练和测试形态识别。
"""
from __future__ import annotations

import json
import time
import random
from pathlib import Path
from statistics import mean
from datetime import datetime

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from trading_system.binance import BinanceFuturesClient
from trading_system.kline_pattern import (
    compute_pattern_features, detect_patterns,
    PATTERN_NAMES, ALL_FEATURE_KEYS, PatternResult,
)


SYMBOLS = ["BUSDT", "RAVEUSDT", "TRBUSDT", "LABUSDT", "ASTERUSDT"]
INTERVALS = ["5m", "15m", "1h", "2h", "4h", "1d"]

# V4 pipeline 实际交易过的币种（从 paper_state journal 提取）
V4_SYMBOLS = [
    "AGTUSDT", "AINUSDT", "AIOTUSDT", "BILLUSDT", "BRUSDT", "BSBUSDT",
    "BTCUSDT", "BUSDT", "COLLECTUSDT", "DAMUSDT", "DASHUSDT", "DOGEUSDT",
    "ESPORTSUSDT", "EVAAUSDT", "FHEUSDT", "GTCUSDT", "HUSDT", "INXUSDT",
    "IRYSUSDT", "JCTUSDT", "LABUSDT", "LAYERUSDT", "MAGMAUSDT", "ONDOUSDT",
    "PLAYUSDT", "QUSDT", "RIVERUSDT", "SAGAUSDT", "SAHARAUSDT", "SKYAIUSDT",
    "SPORTFUNUSDT", "SUIUSDT", "TAGUSDT", "TONUSDT", "TRUTHUSDT", "UBUSDT",
    "USELESSUSDT", "USUSDT", "VELVETUSDT", "ZBTUSDT", "ZECUSDT", "ZEREBROUSDT",
]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def fetch_symbol_data(client: BinanceFuturesClient, symbol: str) -> dict[str, list]:
    """拉取单个币所有时间周期的 kline 数据。"""
    data: dict[str, list] = {}
    for interval in INTERVALS:
        try:
            klines = client.get_json("/fapi/v1/klines", {
                "symbol": symbol, "interval": interval, "limit": 1000,
            })
            data[interval] = klines or []
            log(f"  {symbol}/{interval}: {len(data[interval])} 根K线")
        except Exception as e:
            log(f"  {symbol}/{interval}: 失败 - {e}")
            data[interval] = []
        time.sleep(0.1)
    return data


def find_max_rally(klines: list) -> dict:
    """找到最大涨幅段（连续上涨或短期内最大涨幅）。"""
    if len(klines) < 10:
        return {"start_idx": 0, "end_idx": len(klines) - 1, "pct": 0}

    best_pct = 0
    best_start = 0
    best_end = 0

    # 滑动窗口找最大涨幅（窗口 5~30 根）
    for window in [5, 10, 15, 20, 30]:
        for i in range(len(klines) - window):
            start_price = float(klines[i][1])  # open
            end_price = float(klines[i + window][4])  # close
            if start_price > 0:
                pct = (end_price - start_price) / start_price
                if pct > best_pct:
                    best_pct = pct
                    best_start = i
                    best_end = i + window

    return {"start_idx": best_start, "end_idx": best_end, "pct": best_pct}


def find_max_drop(klines: list) -> dict:
    """找到最大跌幅段。"""
    if len(klines) < 10:
        return {"start_idx": 0, "end_idx": len(klines) - 1, "pct": 0}

    best_pct = 0
    best_start = 0
    best_end = 0

    for window in [5, 10, 15, 20, 30]:
        for i in range(len(klines) - window):
            start_price = float(klines[i][1])
            end_price = float(klines[i + window][4])
            if start_price > 0:
                pct = (start_price - end_price) / start_price  # 跌幅为正数
                if pct > best_pct:
                    best_pct = pct
                    best_start = i
                    best_end = i + window

    return {"start_idx": best_start, "end_idx": best_end, "pct": best_pct}


def _to_kline(item):
    return [int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5])]


def build_samples_around_rally(klines_raw: list, rally: dict, symbol: str, interval: str) -> list[dict]:
    """在最大涨幅段前后构建训练样本。"""
    samples = []
    window_size = 20

    # 取涨幅段前 20 根到涨幅段后 20 根的范围
    start = max(0, rally["start_idx"] - 20)
    end = min(len(klines_raw) - 13, rally["end_idx"] + 20)

    if end - start < window_size + 12:
        return samples

    klines = [_to_kline(k) for k in klines_raw]

    for i in range(max(start, window_size), end):
        window = klines[i - window_size: i + 1]
        close_now = float(klines[i][4])
        if close_now <= 0:
            continue

        features = compute_pattern_features(window)
        features["_symbol"] = symbol
        features["_interval"] = interval
        features["_idx"] = i

        # Forward return
        if i + 3 < len(klines):
            c3 = float(klines[i + 3][4])
            features["_fwd_ret_3"] = round((c3 - close_now) / close_now, 6)
        if i + 12 < len(klines):
            c12 = float(klines[i + 12][4])
            features["_fwd_ret_12"] = round((c12 - close_now) / close_now, 6)

        samples.append(features)

    return samples


def _train_and_report(all_samples: list[dict], label: str) -> dict:
    """对一组样本训练+考试，输出详细日志。返回汇总结果。"""
    if len(all_samples) < 50:
        log(f"  {label}: 样本不足({len(all_samples)})，跳过")
        return {}

    log(f"\n{'─'*50}")
    log(f"训练集: {label} | 总样本: {len(all_samples)}")
    log(f"特征维度: {len(ALL_FEATURE_KEYS)} 个")

    rng = random.Random(42)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    split = int(len(indices) * 0.75)
    train_idx = indices[:split]
    test_idx = indices[split:]
    log(f"训练集: {len(train_idx)} | 测试集: {len(test_idx)}")

    X_train = np.array([[all_samples[i].get(k, 0.0) for k in ALL_FEATURE_KEYS] for i in train_idx])
    X_test = np.array([[all_samples[i].get(k, 0.0) for k in ALL_FEATURE_KEYS] for i in test_idx])

    def _get_dir(s):
        bc = s.get("pattern_bullish_count", 0)
        brc = s.get("pattern_bearish_count", 0)
        if bc > brc: return 1
        elif brc > bc: return -1
        return 0

    y_train_dir = np.array([_get_dir(all_samples[i]) for i in train_idx])
    y_test_dir = np.array([_get_dir(all_samples[i]) for i in test_idx])

    log("\n训练方向分类器...")
    clf_dir = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf_dir.fit(X_train, y_train_dir)
    pred_dir = clf_dir.predict(X_test)
    dir_acc = float(np.mean(pred_dir == y_test_dir))
    log(f"  方向准确率: {dir_acc:.1%}")

    log("\n训练各形态分类器:")
    results: dict[str, dict] = {}
    for pname in PATTERN_NAMES:
        key = f"pat_{pname}"
        y_train = np.array([int(all_samples[i].get(key, 0) > 0.5) for i in train_idx])
        y_test = np.array([int(all_samples[i].get(key, 0) > 0.5) for i in test_idx])
        pos_train = int(y_train.sum())
        pos_test = int(y_test.sum())

        if pos_train < 5 or pos_test < 2:
            results[pname] = {"skipped": True, "train_pos": pos_train, "test_pos": pos_test}
            continue

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        tn = int(((pred == 0) & (y_test == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_test)

        status = "✅" if f1 >= 0.8 else ("⚠️" if f1 >= 0.5 else "❌")
        log(f"  {pname:30s} {status} f1={f1:.3f} prec={precision:.3f} rec={recall:.3f} (n={pos_test})")
        results[pname] = {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4), "support_test": pos_test}

    trained = [(k, v) for k, v in results.items() if not v.get("skipped")]
    skipped = [(k, v) for k, v in results.items() if v.get("skipped")]
    avg_f1 = mean([v["f1"] for _, v in trained]) if trained else 0
    above_80 = sum(1 for _, v in trained if v["f1"] >= 0.8)

    log(f"\n  汇总: 方向={dir_acc:.1%} | 平均F1={avg_f1:.1%} | ≥80%: {above_80}/{len(trained)} | 跳过: {len(skipped)}")
    return {"direction_accuracy": dir_acc, "avg_f1": avg_f1, "above_80": above_80, "trained": len(trained), "skipped": len(skipped), "per_pattern": results}


def run():
    log("=" * 60)
    log("形态识别对比考试: 涨幅段 vs 跌幅段")
    log(f"币种: {SYMBOLS}")
    log(f"时间周期: {INTERVALS}")
    log("=" * 60)

    client = BinanceFuturesClient("https://fapi.binance.com")

    # Phase 1: 拉数据
    log("\n>>> Phase 1: 拉取数据")
    all_data: dict[str, dict[str, list]] = {}
    for symbol in SYMBOLS:
        log(f"\n拉取 {symbol}:")
        all_data[symbol] = fetch_symbol_data(client, symbol)

    # Phase 2: 找涨幅段和跌幅段
    log("\n>>> Phase 2: 寻找历史最大涨幅段 & 最大跌幅段")
    rallies: dict[str, dict[str, dict]] = {}
    drops: dict[str, dict[str, dict]] = {}
    for symbol in SYMBOLS:
        rallies[symbol] = {}
        drops[symbol] = {}
        log(f"\n{symbol}:")
        for interval in INTERVALS:
            klines = all_data[symbol].get(interval, [])
            if len(klines) < 20:
                continue
            rally = find_max_rally(klines)
            drop = find_max_drop(klines)
            rallies[symbol][interval] = rally
            drops[symbol][interval] = drop

            if rally["pct"] > 0:
                start_ts = int(klines[rally["start_idx"]][0]) / 1000
                end_ts = int(klines[rally["end_idx"]][0]) / 1000
                log(f"  {interval:4s} 涨: +{rally['pct']*100:.1f}% ({datetime.fromtimestamp(start_ts).strftime('%m-%d %H:%M')} → {datetime.fromtimestamp(end_ts).strftime('%m-%d %H:%M')})")
            if drop["pct"] > 0:
                start_ts = int(klines[drop["start_idx"]][0]) / 1000
                end_ts = int(klines[drop["end_idx"]][0]) / 1000
                log(f"  {interval:4s} 跌: -{drop['pct']*100:.1f}% ({datetime.fromtimestamp(start_ts).strftime('%m-%d %H:%M')} → {datetime.fromtimestamp(end_ts).strftime('%m-%d %H:%M')})")

    # Phase 3: 分别构建涨幅段和跌幅段样本
    log("\n>>> Phase 3: 构建样本")

    rally_samples: list[dict] = []
    drop_samples: list[dict] = []

    log("\n--- 涨幅段样本 ---")
    for symbol in SYMBOLS:
        count = 0
        for interval in INTERVALS:
            klines = all_data[symbol].get(interval, [])
            rally = rallies.get(symbol, {}).get(interval)
            if not rally or len(klines) < 40:
                continue
            samples = build_samples_around_rally(klines, rally, symbol, interval)
            rally_samples.extend(samples)
            count += len(samples)
            if samples:
                detected: dict[str, int] = {}
                for s in samples:
                    for pname in PATTERN_NAMES:
                        if s.get(f"pat_{pname}", 0) > 0.5:
                            detected[pname] = detected.get(pname, 0) + 1
                top = sorted(detected.items(), key=lambda x: -x[1])[:4]
                log(f"  {symbol}/{interval}: {len(samples)}样本 | {', '.join(f'{p}({c})' for p,c in top)}")
        log(f"  {symbol} 涨幅段合计: {count}")

    log("\n--- 跌幅段样本 ---")
    for symbol in SYMBOLS:
        count = 0
        for interval in INTERVALS:
            klines = all_data[symbol].get(interval, [])
            drop = drops.get(symbol, {}).get(interval)
            if not drop or len(klines) < 40:
                continue
            samples = build_samples_around_rally(klines, drop, symbol, interval)
            drop_samples.extend(samples)
            count += len(samples)
            if samples:
                detected: dict[str, int] = {}
                for s in samples:
                    for pname in PATTERN_NAMES:
                        if s.get(f"pat_{pname}", 0) > 0.5:
                            detected[pname] = detected.get(pname, 0) + 1
                top = sorted(detected.items(), key=lambda x: -x[1])[:4]
                log(f"  {symbol}/{interval}: {len(samples)}样本 | {', '.join(f'{p}({c})' for p,c in top)}")
        log(f"  {symbol} 跌幅段合计: {count}")

    log(f"\n涨幅段总样本: {len(rally_samples)} | 跌幅段总样本: {len(drop_samples)}")

    # Phase 4: 分别训练+考试
    log("\n>>> Phase 4: 分别训练 & 考试")

    log("\n" + "=" * 60)
    log("【A】涨幅段训练")
    log("=" * 60)
    rally_result = _train_and_report(rally_samples, "涨幅段")

    log("\n" + "=" * 60)
    log("【B】跌幅段训练")
    log("=" * 60)
    drop_result = _train_and_report(drop_samples, "跌幅段")

    # Phase 5: 混合训练（涨+跌合并）
    log("\n" + "=" * 60)
    log("【C】混合训练（涨幅+跌幅合并）")
    log("=" * 60)
    mixed_samples = rally_samples + drop_samples
    mixed_result = _train_and_report(mixed_samples, "混合")

    # Phase 6: 对比总结
    log("\n" + "=" * 60)
    log(">>> Phase 5: 对比总结")
    log("=" * 60)
    log(f"{'指标':<20s} {'涨幅段':>10s} {'跌幅段':>10s} {'混合':>10s}")
    log(f"{'─'*50}")
    if rally_result and drop_result and mixed_result:
        log(f"{'方向准确率':<20s} {rally_result['direction_accuracy']:>9.1%} {drop_result['direction_accuracy']:>9.1%} {mixed_result['direction_accuracy']:>9.1%}")
        log(f"{'平均 F1':<20s} {rally_result['avg_f1']:>9.1%} {drop_result['avg_f1']:>9.1%} {mixed_result['avg_f1']:>9.1%}")
        log(f"{'F1≥80% 形态数':<20s} {rally_result['above_80']:>10d} {drop_result['above_80']:>10d} {mixed_result['above_80']:>10d}")
        log(f"{'可训练形态数':<20s} {rally_result['trained']:>10d} {drop_result['trained']:>10d} {mixed_result['trained']:>10d}")

        # 逐形态对比
        log(f"\n{'形态':<28s} {'涨F1':>8s} {'跌F1':>8s} {'混合F1':>8s}")
        log(f"{'─'*56}")
        all_patterns = set()
        for r in [rally_result, drop_result, mixed_result]:
            for k, v in r.get("per_pattern", {}).items():
                if not v.get("skipped"):
                    all_patterns.add(k)
        for pname in sorted(all_patterns):
            r_f1 = rally_result.get("per_pattern", {}).get(pname, {}).get("f1", "-")
            d_f1 = drop_result.get("per_pattern", {}).get(pname, {}).get("f1", "-")
            m_f1 = mixed_result.get("per_pattern", {}).get(pname, {}).get("f1", "-")
            r_str = f"{r_f1:.3f}" if isinstance(r_f1, float) else "  skip"
            d_str = f"{d_f1:.3f}" if isinstance(d_f1, float) else "  skip"
            m_str = f"{m_f1:.3f}" if isinstance(m_f1, float) else "  skip"
            log(f"  {pname:<26s} {r_str:>8s} {d_str:>8s} {m_str:>8s}")

    log("\n完成。")


def run_full():
    """全量数据训练：5个币 × 6个时间周期 × 全部K线。"""
    log("=" * 60)
    log("全量形态识别训练 - 增强特征版")
    log(f"币种: {SYMBOLS}")
    log(f"时间周期: {INTERVALS}")
    log(f"特征维度: {len(ALL_FEATURE_KEYS)} 个")
    log("=" * 60)

    client = BinanceFuturesClient("https://fapi.binance.com")

    log("\n>>> Phase 1: 拉取全量数据 (每周期1000根)")
    all_data: dict[str, dict[str, list]] = {}
    for symbol in SYMBOLS:
        log(f"\n{symbol}:")
        all_data[symbol] = fetch_symbol_data(client, symbol)

    log("\n>>> Phase 2: 构建全量训练样本")
    all_samples: list[dict] = []
    window_size = 20

    for symbol in SYMBOLS:
        sym_count = 0
        for interval in INTERVALS:
            klines_raw = all_data[symbol].get(interval, [])
            if len(klines_raw) < window_size + 12:
                continue
            klines = [_to_kline(k) for k in klines_raw]
            interval_count = 0
            for i in range(window_size, len(klines) - 12):
                window = klines[i - window_size: i + 1]
                close_now = float(klines[i][4])
                if close_now <= 0:
                    continue
                features = compute_pattern_features(window)
                features["_symbol"] = symbol
                features["_interval"] = interval
                all_samples.append(features)
                interval_count += 1

            detected: dict[str, int] = {}
            for s in all_samples[-interval_count:]:
                for pname in PATTERN_NAMES:
                    if s.get(f"pat_{pname}", 0) > 0.5:
                        detected[pname] = detected.get(pname, 0) + 1
            top = sorted(detected.items(), key=lambda x: -x[1])[:5]
            pat_str = ", ".join(f"{p}({c})" for p, c in top)
            log(f"  {symbol}/{interval}: {interval_count} 样本 | {pat_str}")
            sym_count += interval_count
        log(f"  {symbol} 合计: {sym_count}")

    log(f"\n总样本: {len(all_samples)}")

    log("\n>>> 形态分布:")
    total_patterns: dict[str, int] = {}
    for s in all_samples:
        for pname in PATTERN_NAMES:
            if s.get(f"pat_{pname}", 0) > 0.5:
                total_patterns[pname] = total_patterns.get(pname, 0) + 1
    for p, c in sorted(total_patterns.items(), key=lambda x: -x[1]):
        pct = c / len(all_samples) * 100
        log(f"  {p:30s}: {c:5d} ({pct:.1f}%)")

    log("\n>>> Phase 3: 训练 GBM (增强特征)")
    _train_and_report(all_samples, "全量")
    log("\n完成。")


def run_v4():
    """V4 币种全量训练：42个币 × 6个时间周期，针对4个弱形态。"""
    target_weak = ["dark_cloud_cover", "piercing_line", "morning_star_after_decline", "evening_star_after_rally"]

    log("=" * 60)
    log("V4 币种针对性训练 — 重点: 4个弱形态")
    log(f"目标形态: {target_weak}")
    log(f"币种数: {len(V4_SYMBOLS)}")
    log(f"时间周期: {INTERVALS}")
    log(f"特征维度: {len(ALL_FEATURE_KEYS)} 个")
    log("=" * 60)

    client = BinanceFuturesClient("https://fapi.binance.com")

    log("\n>>> Phase 1: 拉取全量数据 (每周期1000根)")
    all_data: dict[str, dict[str, list]] = {}
    valid_symbols: list[str] = []
    for symbol in V4_SYMBOLS:
        log(f"\n{symbol}:")
        data = fetch_symbol_data(client, symbol)
        total_bars = sum(len(v) for v in data.values())
        if total_bars > 0:
            all_data[symbol] = data
            valid_symbols.append(symbol)
        else:
            log(f"  跳过（无数据）")

    log(f"\n有效币种: {len(valid_symbols)}/{len(V4_SYMBOLS)}")

    log("\n>>> Phase 2: 构建全量训练样本")
    all_samples: list[dict] = []
    window_size = 20

    weak_pattern_samples: dict[str, int] = {p: 0 for p in target_weak}

    for symbol in valid_symbols:
        sym_count = 0
        for interval in INTERVALS:
            klines_raw = all_data[symbol].get(interval, [])
            if len(klines_raw) < window_size + 12:
                continue
            klines = [_to_kline(k) for k in klines_raw]
            interval_count = 0
            for i in range(window_size, len(klines) - 12):
                window = klines[i - window_size: i + 1]
                close_now = float(klines[i][4])
                if close_now <= 0:
                    continue
                features = compute_pattern_features(window)
                features["_symbol"] = symbol
                features["_interval"] = interval
                all_samples.append(features)
                interval_count += 1

                for wp in target_weak:
                    if features.get(f"pat_{wp}", 0) > 0.5:
                        weak_pattern_samples[wp] += 1

            if interval_count > 0:
                log(f"  {symbol}/{interval}: {interval_count} 样本")
            sym_count += interval_count
        if sym_count > 0:
            log(f"  {symbol} 合计: {sym_count}")

    log(f"\n总样本: {len(all_samples)}")

    log("\n>>> 弱形态出现统计:")
    for wp, cnt in weak_pattern_samples.items():
        pct = cnt / len(all_samples) * 100 if all_samples else 0
        status = "✅" if cnt >= 50 else ("⚠️" if cnt >= 20 else "❌")
        log(f"  {status} {wp:30s}: {cnt:5d} ({pct:.2f}%)")

    log("\n>>> 全部形态分布:")
    total_patterns: dict[str, int] = {}
    for s in all_samples:
        for pname in PATTERN_NAMES:
            if s.get(f"pat_{pname}", 0) > 0.5:
                total_patterns[pname] = total_patterns.get(pname, 0) + 1
    for p, c in sorted(total_patterns.items(), key=lambda x: -x[1]):
        pct = c / len(all_samples) * 100
        marker = " ← TARGET" if p in target_weak else ""
        log(f"  {p:30s}: {c:5d} ({pct:.1f}%){marker}")

    log("\n>>> Phase 3: 训练 GBM")
    result = _train_and_report(all_samples, "V4全量")

    if result:
        log("\n>>> 弱形态考试结果:")
        for wp in target_weak:
            pr = result.get("per_pattern", {}).get(wp, {})
            if pr.get("skipped"):
                log(f"  ❌ {wp}: 跳过 (train_pos={pr.get('train_pos',0)}, test_pos={pr.get('test_pos',0)})")
            else:
                f1 = pr.get("f1", 0)
                prec = pr.get("precision", 0)
                rec = pr.get("recall", 0)
                status = "✅" if f1 >= 0.8 else ("⚠️" if f1 >= 0.5 else "❌")
                log(f"  {status} {wp:30s}: F1={f1:.3f} prec={prec:.3f} rec={rec:.3f}")

    log("\n完成。")


if __name__ == "__main__":
    import sys
    if "--v4" in sys.argv:
        run_v4()
    elif "--full" in sys.argv:
        run_full()
    else:
        run()
