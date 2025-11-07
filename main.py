import os, json, time, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")

MEXC = "https://api.mexc.com"

# === Dosya ===
HISTORY_FILE = "volume_memory.json"

# === Ayarlar ===
SCAN_LIMIT       = 300
VOL_N            = 10
VOL_RATIO_MIN    = 1.20
VOL_Z_MIN        = 0.8
VOL_RAMP_MIN     = 1.3
ALERT_THRESHOLD  = 4          # ✅ 2 gün içinde gereken sinyal sayısı
WINDOW_HOURS     = 48         # ✅ 48 saat pencere

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def jget(url, params=None, retries=2):
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            time.sleep(0.3)
    return None

def telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print(msg)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg}
        )
    except:
        pass

# =============== COIN LİSTESİ (Hacme göre MEXC) ===============
def get_symbols(limit=SCAN_LIMIT):
    d = jget(f"{MEXC}/api/v3/ticker/24hr")
    if not d:
        return []
    coins = [x for x in d if x.get("symbol","").endswith("USDT")]
    coins.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
    return [c["symbol"] for c in coins[:limit]]

# =============== HACİM TESTİ ===============
def volume_signal(df):
    t = df["turnover"]
    if len(t) < VOL_N + 3:
        return False

    base = t.ewm(span=VOL_N, adjust=False).mean()
    ratio = float(t.iloc[-1] / (base.iloc[-2] + 1e-12))

    roll = t.rolling(VOL_N)
    mu = np.log((roll.mean().iloc[-1] or 1e-12))
    sd = np.log((roll.std().iloc[-1] or 1e-12))
    z = (np.log(t.iloc[-1] + 1e-12) - mu) / (sd + 1e-12)

    ramp = float(t.iloc[-3:].sum() / ((roll.mean().iloc[-1] * 3) + 1e-12))

    if ratio >= VOL_RATIO_MIN or z >= VOL_Z_MIN or ramp >= VOL_RAMP_MIN:
        return True

    return False

# =============== 1 SAATLİK KLINE ===============
def get_kline(sym):
    d = jget(f"{MEXC}/api/v3/klines", {"symbol": sym, "interval": "1h", "limit": 50})
    if not d:
        return None
    df = pd.DataFrame(
        d,
        columns=["t","o","h","l","c","v","qv","n","t1","t2","ig","ib"]
    ).astype(float)
    df.rename(columns={"qv": "turnover"}, inplace=True)
    return df

# =============== HAFIZA OKUMA/YAZMA ===============
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_history(hist):
    with open(HISTORY_FILE, "w") as f:
        json.dump(hist, f)

# =============== ANA BOT ===============
def main():
    symbols = get_symbols(SCAN_LIMIT)
    if not symbols:
        telegram("⚠️ Coin listesi alınamadı (MEXC)")
        return

    history = load_history()
    now = datetime.now().timestamp()

    alerts = []

    for sym in symbols:
        df = get_kline(sym)
        if df is None:
            continue

        if volume_signal(df):
            # sinyal kaydet
            history.setdefault(sym, [])
            history[sym].append(now)

            # son 48 saatin kayıtlarını tut
            history[sym] = [x for x in history[sym] 
                            if x >= now - WINDOW_HOURS * 3600]

            # eşik aşılmış mı?
            if len(history[sym]) >= ALERT_THRESHOLD:
                alerts.append(sym)

        time.sleep(0.05)

    save_history(history)

    if alerts:
        msg = "🔥 *Toplanma Alarmı – 48 Saatte 4+ Sinyal*\n\n"
        for s in alerts:
            msg += f"✅ {s}\n"
        telegram(msg)
    else:
        print("No cluster signals at", ts())


if __name__ == "__main__":
    main()
