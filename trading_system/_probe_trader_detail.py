"""
探针：探索 Binance 聪明钱个人交易记录 API
目标：找到能拿到每个交易者历史开仓/平仓/止盈止损的接口
"""
import requests
import json

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/135.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://www.binance.com",
    "Referer": "https://www.binance.com/zh-CN/smart-money",
})

# 用 BlockchainUSA1 做探针
TRADER_ID = "4814568371709014273"  # 先用之前抓过的 Pickle Cat

# 1. profile 完整字段
print("=== 1. Profile 完整数据 ===")
r = session.get(f"https://www.binance.com/bapi/asset/v1/friendly/future/smart-money/profile?topTraderId={TRADER_ID}", timeout=10)
d = r.json().get("data", {})
for k, v in sorted(d.items()):
    if k != "accounts":
        print(f"  {k}: {v}")

# 2. 尝试各种可能的交易记录接口
print("\n\n=== 2. 尝试交易记录接口 ===")

endpoints = [
    # 订单记录
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/orders?topTraderId={TRADER_ID}&page=1&rows=20",
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/order/list?topTraderId={TRADER_ID}&page=1&rows=20",
    f"https://www.binance.com/bapi/asset/v1/friendly/future/smart-money/profile/orders?topTraderId={TRADER_ID}&page=1&rows=20",
    # 持仓历史
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/positions?topTraderId={TRADER_ID}&page=1&rows=20",
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/position/history?topTraderId={TRADER_ID}&page=1&rows=20",
    f"https://www.binance.com/bapi/asset/v1/friendly/future/smart-money/profile/positions?topTraderId={TRADER_ID}&page=1&rows=20",
    # 交易历史
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/trades?topTraderId={TRADER_ID}&page=1&rows=20",
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/trade/list?topTraderId={TRADER_ID}&page=1&rows=20",
    f"https://www.binance.com/bapi/asset/v1/friendly/future/smart-money/profile/trades?topTraderId={TRADER_ID}&page=1&rows=20",
    # PnL 记录
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/pnl?topTraderId={TRADER_ID}&page=1&rows=20",
    f"https://www.binance.com/bapi/asset/v1/friendly/future/smart-money/profile/pnl?topTraderId={TRADER_ID}&page=1&rows=20",
    # 带时间范围
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/orders?topTraderId={TRADER_ID}&page=1&rows=20&timeRange=30D",
    f"https://www.binance.com/bapi/futures/v1/friendly/future/smart-money/profile/positions?topTraderId={TRADER_ID}&page=1&rows=20&timeRange=30D",
]

for url in endpoints:
    try:
        r = session.get(url, timeout=10)
        d = r.json()
        success = d.get("success", False)
        code = d.get("code", "?")
        data = d.get("data")
        if success and data:
            print(f"\n✅ {url[:100]}")
            print(f"   code={code}, data type={type(data).__name__}")
            if isinstance(data, dict):
                print(f"   keys: {sorted(data.keys())}")
                if "rows" in data:
                    rows = data["rows"]
                    print(f"   total={data.get('total')}, rows={len(rows)}")
                    if rows and isinstance(rows[0], dict):
                        print(f"   row fields: {sorted(rows[0].keys())}")
                        print(f"   first: {json.dumps(rows[0], ensure_ascii=False, default=str)[:500]}")
                elif "list" in data:
                    lst = data["list"]
                    print(f"   list len={len(lst)}")
                    if lst and isinstance(lst[0], dict):
                        print(f"   item fields: {sorted(lst[0].keys())}")
            elif isinstance(data, list) and data:
                print(f"   list len={len(data)}")
                if isinstance(data[0], dict):
                    print(f"   item fields: {sorted(data[0].keys())}")
                    print(f"   first: {json.dumps(data[0], ensure_ascii=False, default=str)[:500]}")
        else:
            msg = d.get("message", "")
            print(f"   ❌ {url[:100]}  code={code} msg={msg}")
    except Exception as e:
        print(f"   ❌ {url[:100]}  err={e}")

# 3. 看看之前 capture 里有没有更多 API 线索
print("\n\n=== 3. 从之前 capture 里找 profile 页面的 API ===")
import os
capture_path = "data/trading_system/binance_smart_money_capture.json"
if os.path.exists(capture_path):
    with open(capture_path) as f:
        cap_data = json.load(f)
    for cap in cap_data.get("captures", []):
        if cap.get("label") == "profile":
            for r in cap.get("records", []):
                url = r.get("url", "")
                if "bapi" in url and "profile" in url:
                    text = r.get("text", "")
                    if text and text.startswith("{"):
                        try:
                            d = json.loads(text)
                            if d.get("success") and d.get("data"):
                                print(f"\n  URL: {url[:120]}")
                                data = d["data"]
                                if isinstance(data, dict):
                                    print(f"  keys: {sorted(data.keys())}")
                        except:
                            pass
