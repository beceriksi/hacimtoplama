import os, time, math, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# === Secrets (GitHub → Settings → Secrets and variables → Actions) ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")

# === Sabitler ===
OKX_BASE       = "https://www.okx.com"
PAIR_LIST      = ["BTC-USDT", "ETH-USDT"]      # Sadece BTC & ETH — az gürültü, yüksek kalite
REPORT_TFS     = ["15m", "1H", "4H"]           # Confluence için bakılan TF’ler
SCHEDULE_BAR   = "1H"                          # 48h balina akışı bu TF üzerinden ölçülür

# — Whale parametreleri (gerekirse workflow env ile override edebilirsin) —
W_MIN_USD      = float(os.getenv("W_MIN_USD", "1500000"))   # tek mum “balina” sayılacak asgari USD
W_EMA_N        = int(os.getenv("W_EMA_N", "12"))            # hacim-EMA
W_RATIO        = float(os.getenv("W_RATIO", "1.35"))        # EMA-1’e oran eşik
W_DECAY_HRS    = int(os.getenv("W_DECAY_HRS", "36"))        # yakın geçmişe daha çok ağırlık
# — Sinyal eşikleri —
RSI_BUY        = float(os.getenv("RSI_BUY", "55.0"))
RSI_SELL       = float(os.getenv("RSI_SELL", "45.0"))
ADX_BUY        = float(os.getenv("ADX_BUY", "20.0"))
ADX_SELL       = float(os.getenv("ADX_SELL", "18.0"))
NEAR_THR       = float(os.getenv("NEAR_THR", "0.4"))        # tepe/dip yakınlık
MIN_CONF       = int(os.getenv("MIN_CONF", "75"))           # min güven puanı

# === Yardımcılar ===
def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def jget(url, params=None, retries=3, timeout=12):
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except:
            time.sleep(0.4)
    return None

def telegram(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print(text); return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=15)
    except: pass

def ema(x, n): return x.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / (dn.ewm(alpha=1/n, adjust=False).mean() + 1e-12)
    return 100 - (100/(1+rs))

def adx_from_hlc(h,l,c, n=14):
    up = h.diff(); dn = -l.diff()
    plus  = np.where((up>dn)&(up>0), up, 0.0)
    minus = np.where((dn>up)&(dn>0), dn, 0.0)
    tr1 = h - l
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr  = pd.DataFrame({'a':tr1, 'b':tr2, 'c':tr3}).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100*pd.Series(plus ).ewm(alpha=1/n, adjust=False).mean()/(atr+1e-12)
    minus_di = 100*pd.Series(minus).ewm(alpha=1/n, adjust=False).mean()/(atr+1e-12)
    dx  = ((plus_di - minus_di).abs()/((plus_di + minus_di)+1e-12))*100
    return dx.ewm(alpha=1/n, adjust=False).mean()

def near_high(o,h,l,c,th=NEAR_THR):
    rng  = max(h-l, 1e-12)
    body = abs(c-o)/rng
    return (c>o) and ((h-c)/rng <= th) and (body >= 0.30)

def near_low(o,h,l,c,th=NEAR_THR):
    rng  = max(h-l, 1e-12)
    body = abs(c-o)/rng
    return (c<o) and ((c-l)/rng <= th) and (body >= 0.30)

# === OKX Candles ===
def okx_candles(instId, bar="1H", limit=300):
    # OKX v5 market/candles
    # data: [ts, o, h, l, c, vol, volCcy, volCcyQuote?, confirm]
    j = jget(f"{OKX_BASE}/api/v5/market/candles", {"instId":instId, "bar":bar, "limit":limit})
    if not j or j.get("code")!='0' or "data" not in j: return None
    rows = j["data"]
    # OKX sıralamayı ters veriyor; kronolojik yapalım
    rows = rows[::-1]
    df = pd.DataFrame(rows, columns=["ts","o","h","l","c","vol","volCcy","volQ","confirm"])
    for col in ["o","h","l","c","vol","volCcy","volQ"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # USD ciro için volQ (quote) varsa onu kullan
    df["turnover"] = df["volQ"].fillna(df["volCcy"])
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close"}, inplace=True)
    return df[["open","high","low","close","turnover"]]

# === Whale skoru (48 saat, 1H) ===
def whale_pressure(instId, hours=48, bar=SCHEDULE_BAR):
    df = okx_candles(instId, bar=bar, limit=hours+10)
    if df is None or len(df) < 20: return {"score":0, "buys":0, "sells":0, "spikes":0}
    t  = df["turnover"]; c = df["close"]; o = df["open"]
    vr = t / (ema(t, W_EMA_N).shift(1) + 1e-12)

    spikes = []
    # son ‘hours’ barı değerlendir
    look = min(hours, len(df)-1)
    for i in range(len(df)-look, len(df)):
        usd   = float(t.iloc[i])
        ratio = float(vr.iloc[i])
        if usd >= W_MIN_USD and ratio >= W_RATIO:
            # yakın barlara daha fazla ağırlık (exp-decay)
            age = (len(df)-1) - i
            decay = math.exp(-age / max(1, W_DECAY_HRS))
            side = 1 if c.iloc[i] > o.iloc[i] else -1
            spikes.append((side, decay, usd, ratio))

    if not spikes:
        return {"score":0, "buys":0, "sells":0, "spikes":0}

    buys  = sum(1 for s,_,_,_ in spikes if s==1)
    sells = sum(1 for s,_,_,_ in spikes if s==-1)
    score = 0.0
    for s,dec,usd,ratio in spikes:
        unit = (min(3.0, ratio) - 1.0) * 50.0   # 1.35 → ~17.5 puan
        unit += min(40.0, usd / 5_000_000.0)    # 5M $ → +1 puan, 200M $ cap
        unit *= dec
        score += (unit if s==1 else -unit)

    # normalize 0–100 aralığına (asimetrik olabilir; sadeleştirme)
    norm = max(-200.0, min(200.0, score))
    final = int(50 + (norm/2.0))
    return {"score":max(0, min(100, final)), "buys":buys, "sells":sells, "spikes":len(spikes)}

# === Confluence sinyali (az ama öz) ===
def confluence_signal(instId):
    out = {"pair":instId, "side":None, "conf":0, "lines":[]}
    votes_buy = 0; votes_sell = 0; details = []

    for tf in REPORT_TFS:
        df = okx_candles(instId, bar=tf, limit=180)
        if df is None or len(df) < 60:
            details.append(f"{instId} {tf}: veri yok"); continue
        o,h,l,c = df["open"], df["high"], df["low"], df["close"]
        e20 = float(ema(c,20).iloc[-1]); e50 = float(ema(c,50).iloc[-1])
        trend_up = e20 > e50
        rr  = float(rsi(c,14).iloc[-1])
        adxv= float(adx_from_hlc(h,l,c,14).iloc[-1])
        # Hacim oranı
        vr  = df["turnover"] / (ema(df["turnover"], 12).shift(1) + 1e-12)
        vrl = float(vr.iloc[-1])
        # Tepe/dip yakınlığı
        nh  = near_high(o.iloc[-1],h.iloc[-1],l.iloc[-1],c.iloc[-1])
        nl  = near_low (o.iloc[-1],h.iloc[-1],l.iloc[-1],c.iloc[-1])

        buy_ok  = trend_up and rr>=RSI_BUY  and adxv>=ADX_BUY  and vrl>=1.20 and nh
        sell_ok = (not trend_up) and rr<=RSI_SELL and adxv>=ADX_SELL and vrl>=1.10 and nl

        if buy_ok:  votes_buy  += 1
        if sell_ok: votes_sell += 1

        details.append(f"{instId} {tf}: trend={'↑' if trend_up else '↓'} RSI:{rr:.1f} ADX:{adxv:.0f} v:{vrl:.2f} nh/nl:{'T' if nh else '-'}{'B' if nl else '-'}")

    # Oy sayımı
    side = None
    if votes_buy >= 2 and votes_buy > votes_sell:
        side = "BUY"
    elif votes_sell >= 2 and votes_sell > votes_buy:
        side = "SELL"

    if side:
        # Güven puanı: TF sayısı + whale bias
        whale = whale_pressure(instId, hours=48, bar="1H")
        base  = 30 + 25*max(votes_buy, votes_sell)    # 2 TF → 80, 3 TF → 105 (sonra clamp)
        bias  = (whale["score"]-50)/2.0               # -25…+25
        conf  = int(max(0, min(100, base + bias)))
        out.update({"side":side, "conf":conf})
    out["lines"] = details
    return out

# === Ana akış ===
def main():
    # 48 saatlik whale resmi
    btc_w = whale_pressure("BTC-USDT", hours=48, bar=SCHEDULE_BAR)
    eth_w = whale_pressure("ETH-USDT", hours=48, bar=SCHEDULE_BAR)

    # ETH vs BTC kıyası
    verdict = "→ Dengeli"
    diff = eth_w["score"] - btc_w["score"]
    if diff >= 8:  verdict = "🟢 *ETH lehine balina akışı*"
    if diff <= -8: verdict = "🔵 *BTC lehine balina akışı*"

    # Confluence sinyali (az ama öz)
    btc_sig = confluence_signal("BTC-USDT")
    eth_sig = confluence_signal("ETH-USDT")

    lines = [
        f"🧭 *Saatlik Özet* — {ts()}",
        f"📊 48h Whale Skorları (1H): BTC:{btc_w['score']}/100 (spike:{btc_w['spikes']} | B:{btc_w['buys']} S:{btc_w['sells']}) | ETH:{eth_w['score']}/100 (spike:{eth_w['spikes']} | B:{eth_w['buys']} S:{eth_w['sells']})",
        f"⚖️ Piyasa Fikri: {verdict}",
    ]

    def fmt_sig(s):
        if not s["side"]: return f"{s['pair']}: sinyal yok"
        tag  = "🟢 BUY" if s["side"]=="BUY" else "🔴 SELL"
        mark = " ✅" if s["conf"]>=MIN_CONF else " (zayıf)"
        return f"{s['pair']}: {tag} | Güven:{s['conf']}{mark}"

    lines.append("\n🎯 *Confluence Sinyal* (15m/1H/4H)")
    lines.append("• " + fmt_sig(btc_sig))
    lines.append("• " + fmt_sig(eth_sig))

    # Ayrıntı (diagnostics) — spam olmasın diye minimal tutuyoruz
    for s in [btc_sig, eth_sig]:
        if s["side"] and s["conf"]>=MIN_CONF:
            lines.append(f"\n— *{s['pair']} detay* —")
            lines.extend([f"  {ln}" for ln in s["lines"]])

    telegram("\n".join(lines))

if __name__ == "__main__":
    main()
