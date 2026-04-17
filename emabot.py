import ccxt
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import asyncio
import io
import os
import time
import csv
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ================= CONFIG =================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
SUBSCRIBERS_FILE = "subscribers.txt"
SIGNAL_LOG_FILE  = "signal_log_ema.csv"

# Timeframes
TIMEFRAMES = ["15m", "30m", "1h"]

# Higher timeframe required to be bullish before lower TF fires
HTF_REQUIRED = {
    "15m": "1h",   # 15m signal only fires if 1h is bullish
    "30m": "4h",   # 30m signal only fires if 4h is bullish
    "1h":  "1d",   # 1h signal only fires if 1d is bullish
}

# Top movers to scan
TOP_MOVERS_COUNT = 50

# Scan interval
SCAN_INTERVAL = 60

# Cooldown per signal (seconds)
SIGNAL_COOLDOWN = {
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
}

# Minimum consecutive red candles before EMA pullback signal
MIN_PULLBACK_CANDLES = 2

# Max pullback depth before considered trend break
MAX_PULLBACK_PCT = {
    "15m": 3.0,
    "30m": 4.0,
    "1h":  5.0,
}

# EMA settings
EMA_FAST = 20
EMA_SLOW = 50

# Pullback must come within this % of EMA to count as touch
EMA_TOUCH_PCT = 2.0

# Candle quality: upper wick cannot be longer than body * this value
# e.g. 1.5 means upper wick max = 1.5x body size
MAX_UPPER_WICK_RATIO = 1.5

# RSI range
RSI_MIN = 38
RSI_MAX = 65

# Volume multiplier vs 20-candle average
VOL_MULT = 1.2

# TP per timeframe (%)
TP_PCT = {
    "15m": 2.0,
    "30m": 3.0,
    "1h":  4.5,
}

# SL buffer below swing low (%)
SL_BUFFER_PCT = 0.2

# Minimum R:R
MIN_RR = 1.8

# Higher High Breakout Retest settings
HH_SWING_LOOKBACK  = 30   # candles to look back for swing high
HH_BREAKOUT_MIN    = 1.5  # price must break above swing high by at least this %
HH_RETEST_ZONE_PCT = 1.2  # retest must come within this % of broken level
HH_COOLDOWN        = {
    "15m": 1800,
    "30m": 3600,
    "1h":  7200,
}
# ==========================================

exchange = ccxt.bybit({
    "enableRateLimit": True,
    "rateLimit": 200,        # ms between requests — Bybit allows 10/s, we use 5/s to be safe
    "options": {"defaultType": "linear"}
})

ALERT_MEMORY  = {}
HTF_CACHE     = {}   # (symbol, htf) -> (trend, timestamp)
HTF_CACHE_TTL = 600  # 10 minutes
_scan_running = False


# ================= SAFE FETCH =================
def fetch_ohlcv_safe(symbol, tf, limit=120, retries=3):
    """
    Fetches OHLCV with automatic retry on rate limit.
    Waits 10s on first hit, 20s on second, 30s on third.
    Returns DataFrame or None.
    """
    for attempt in range(retries):
        try:
            raw = exchange.fetch_ohlcv(symbol, tf, limit=limit)
            time.sleep(0.5)   # 500ms pause after every successful fetch
            if len(raw) < 10:
                return None
            return pd.DataFrame(raw, columns=["time","open","high","low","close","vol"])
        except Exception as e:
            err = str(e)
            if "10006" in err or "Rate Limit" in err or "Too many" in err:
                wait = (attempt + 1) * 10
                log(f"⚠️ Rate limit on {symbol} {tf} — waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                # Non-rate-limit error, don't retry
                return None
    return None


# ================= UTILS =================
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def read_subscribers():
    if not os.path.exists(SUBSCRIBERS_FILE):
        return []
    with open(SUBSCRIBERS_FILE) as f:
        return [x.strip() for x in f if x.strip()]

def save_subscriber(chat_id):
    if str(chat_id) not in read_subscribers():
        with open(SUBSCRIBERS_FILE, "a") as f:
            f.write(str(chat_id) + "\n")

def remove_subscriber(chat_id):
    subs = read_subscribers()
    with open(SUBSCRIBERS_FILE, "w") as f:
        for s in subs:
            if s != str(chat_id):
                f.write(s + "\n")

def cooldown_ok(key, seconds):
    return key not in ALERT_MEMORY or time.time() - ALERT_MEMORY[key] > seconds

def mark_cooldown(key):
    ALERT_MEMORY[key] = time.time()

def clean_sym(s):
    return s.replace(":USDT", "").replace("/USDT", "")

def fmt_price(v):
    if v >= 1000:    return f"{v:,.2f}"
    elif v >= 1:     return f"{v:.4f}"
    elif v >= 0.01:  return f"{v:.5f}"
    else:            return f"{v:.7f}"

def log_signal(symbol, tf, setup, entry, tp, sl, rr, rsi):
    exists = os.path.exists(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","symbol","tf","setup","entry","tp","sl","rr","rsi"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol, tf, setup,
            round(entry, 8), round(tp, 8), round(sl, 8),
            round(rr, 2), round(rsi, 1)
        ])


# ================= INDICATORS =================
def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


# ================= HTF TREND CHECK =================
def get_htf_trend(symbol, htf):
    """
    Returns True if higher timeframe is bullish.
    Uses fetch_ohlcv_safe so rate limits are handled automatically.
    Cached for HTF_CACHE_TTL seconds.
    """
    cache_key = (symbol, htf)
    now = time.time()
    if cache_key in HTF_CACHE:
        trend, ts = HTF_CACHE[cache_key]
        if now - ts < HTF_CACHE_TTL:
            return trend

    df = fetch_ohlcv_safe(symbol, htf, limit=60)
    if df is None or len(df) < 55:
        # Reuse stale cache on failure rather than defaulting to False
        if cache_key in HTF_CACHE:
            return HTF_CACHE[cache_key][0]
        HTF_CACHE[cache_key] = (False, now)
        return False

    ema50  = compute_ema(df["close"], EMA_SLOW)
    price  = df["close"].iloc[-2]
    e_now  = ema50.iloc[-2]
    e_5ago = ema50.iloc[-5]

    bullish = (
        not pd.isna(e_now) and
        not pd.isna(e_5ago) and
        price > e_now and
        e_now > e_5ago
    )
    HTF_CACHE[cache_key] = (bullish, now)
    return bullish


# ================= CANDLE QUALITY CHECK =================
def candle_quality_ok(candle):
    """
    Returns True if signal candle is clean:
    - Upper wick must not exceed MAX_UPPER_WICK_RATIO * body
      (long upper wick = sellers pushed back hard = weak signal)
    - Body must be at least 30% of total range (no doji)
    """
    body       = abs(candle["close"] - candle["open"])
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    total      = candle["high"] - candle["low"]

    if total == 0:
        return False
    if body / total < 0.30:
        return False
    if body > 0 and upper_wick > body * MAX_UPPER_WICK_RATIO:
        return False

    return True


# ================= EMA PULLBACK DETECTION =================
def detect_ema_pullback(df, tf, htf_bullish):
    """
    EMA Pullback in uptrend with all upgrades:
    - HTF must be bullish
    - Candle quality check (no long upper wick, no doji)
    - All original conditions
    """
    if len(df) < 60:
        return None

    if not htf_bullish:
        return None

    df = df.copy()
    df["ema20"] = compute_ema(df["close"], EMA_FAST)
    df["ema50"] = compute_ema(df["close"], EMA_SLOW)
    df["rsi"]   = compute_rsi(df["close"])

    sig_idx    = -2
    sig_candle = df.iloc[sig_idx]
    ema20_val  = sig_candle["ema20"]
    ema50_val  = sig_candle["ema50"]

    if pd.isna(ema20_val) or pd.isna(ema50_val):
        return None

    # EMA50 sloping up
    ema50_3ago = df["ema50"].iloc[sig_idx - 3]
    if pd.isna(ema50_3ago) or ema50_val <= ema50_3ago:
        return None

    # Price above EMA50
    if sig_candle["close"] < ema50_val:
        return None

    # Signal candle bullish
    if sig_candle["close"] <= sig_candle["open"]:
        return None

    # Signal candle closes above EMA20
    if sig_candle["close"] < ema20_val:
        return None

    # ✅ Candle quality check
    if not candle_quality_ok(sig_candle):
        return None

    # Count consecutive red candles before signal
    pullback_candles = 0
    pullback_low     = sig_candle["low"]
    pullback_high    = None

    for i in range(sig_idx - 1, max(sig_idx - 10, -len(df) - 1), -1):
        candle = df.iloc[i]
        if candle["close"] < candle["open"]:
            pullback_candles += 1
            pullback_low = min(pullback_low, candle["low"])
        else:
            pullback_high = candle["high"]
            break

    if pullback_candles < MIN_PULLBACK_CANDLES:
        return None

    # Pullback depth check
    if pullback_high is None:
        pullback_high = df["high"].iloc[sig_idx - pullback_candles - 2]
    if pullback_high > 0:
        depth_pct = (pullback_high - pullback_low) / pullback_high * 100
        if depth_pct > MAX_PULLBACK_PCT.get(tf, 4.0):
            return None

    # EMA touch check
    near_ema20 = abs(pullback_low - ema20_val) / ema20_val * 100 <= EMA_TOUCH_PCT
    near_ema50 = abs(pullback_low - ema50_val) / ema50_val * 100 <= EMA_TOUCH_PCT
    if not (near_ema20 or near_ema50):
        return None
    ema_touched = "EMA20" if near_ema20 else "EMA50"

    # RSI check
    rsi_val = sig_candle["rsi"]
    if pd.isna(rsi_val) or not (RSI_MIN <= rsi_val <= RSI_MAX):
        return None

    # Volume check
    vol_avg = df["vol"].rolling(20).mean().iloc[sig_idx]
    if pd.isna(vol_avg) or vol_avg == 0:
        return None
    if sig_candle["vol"] < vol_avg * VOL_MULT:
        return None

    # Entry / TP / SL
    entry = sig_candle["close"]
    sl    = pullback_low * (1 - SL_BUFFER_PCT / 100)
    tp    = entry * (1 + TP_PCT.get(tf, 3.0) / 100)

    recent_high = df["high"].iloc[sig_idx - 20: sig_idx].max()
    if recent_high > entry * 1.01:
        alt_rr = (recent_high - entry) / (entry - sl) if entry > sl else 0
        tp_rr  = (tp - entry)          / (entry - sl) if entry > sl else 0
        if alt_rr > tp_rr:
            tp = recent_high

    risk   = entry - sl
    reward = tp - entry
    if risk <= 0:
        return None
    rr = reward / risk
    if rr < MIN_RR:
        return None

    return {
        "setup":            "EMA Pullback",
        "entry":            entry,
        "tp":               tp,
        "sl":               sl,
        "rr":               rr,
        "rsi":              rsi_val,
        "pullback_candles": pullback_candles,
        "pullback_low":     pullback_low,
        "ema_touched":      ema_touched,
        "ema20":            ema20_val,
        "ema50":            ema50_val,
        "vol_mult":         sig_candle["vol"] / vol_avg,
        "df":               df,
        "broken_level":     None,
    }


# ================= HIGHER HIGH BREAKOUT RETEST =================
def detect_hh_retest(df, tf, htf_bullish):
    """
    Higher High Breakout Retest:
    1. Find swing high in last HH_SWING_LOOKBACK candles
    2. Price broke above it by at least HH_BREAKOUT_MIN%
    3. Price pulled back and is now retesting that broken level
    4. Signal candle is green + quality check + RSI + volume
    5. HTF must be bullish
    """
    if len(df) < 60:
        return None

    if not htf_bullish:
        return None

    df = df.copy()
    df["ema20"] = compute_ema(df["close"], EMA_FAST)
    df["ema50"] = compute_ema(df["close"], EMA_SLOW)
    df["rsi"]   = compute_rsi(df["close"])

    sig_idx    = -2
    sig_candle = df.iloc[sig_idx]

    # Signal candle must be bullish
    if sig_candle["close"] <= sig_candle["open"]:
        return None

    # Candle quality check
    if not candle_quality_ok(sig_candle):
        return None

    # Find swing high in lookback window before recent candles
    lookback_start = max(0, len(df) + sig_idx - HH_SWING_LOOKBACK)
    lookback_end   = max(0, len(df) + sig_idx - 4)

    if lookback_end <= lookback_start:
        return None

    swing_window   = df.iloc[lookback_start:lookback_end]
    if swing_window.empty:
        return None

    swing_high_val = swing_window["high"].max()
    swing_high_idx = swing_window["high"].idxmax()

    if swing_high_val <= 0:
        return None

    # Breakout: price closed above swing high by HH_BREAKOUT_MIN%
    post_swing     = df.iloc[swing_high_idx + 1: len(df) + sig_idx - 1]
    if post_swing.empty:
        return None

    breakout_close = post_swing["close"].max()
    breakout_pct   = (breakout_close - swing_high_val) / swing_high_val * 100
    if breakout_pct < HH_BREAKOUT_MIN:
        return None

    # Retest: current price is near the broken swing high
    price        = sig_candle["close"]
    distance_pct = abs(price - swing_high_val) / swing_high_val * 100
    if distance_pct > HH_RETEST_ZONE_PCT:
        return None

    # Price approaching from above (not already broken below)
    if price < swing_high_val * 0.995:
        return None

    # Price above EMA50
    ema50_val = sig_candle["ema50"]
    if pd.isna(ema50_val) or price < ema50_val * 0.98:
        return None

    # RSI check
    rsi_val = sig_candle["rsi"]
    if pd.isna(rsi_val) or not (RSI_MIN <= rsi_val <= RSI_MAX):
        return None

    # Volume check
    vol_avg = df["vol"].rolling(20).mean().iloc[sig_idx]
    if pd.isna(vol_avg) or vol_avg == 0:
        return None
    if sig_candle["vol"] < vol_avg * VOL_MULT:
        return None

    # Entry / TP / SL
    entry = sig_candle["close"]
    sl    = swing_high_val * (1 - SL_BUFFER_PCT / 100)

    recent_highs = df["high"].iloc[swing_high_idx:]
    tp_candidate = recent_highs.max()
    tp = tp_candidate if tp_candidate > entry * 1.005 \
         else entry * (1 + TP_PCT.get(tf, 3.0) / 100)

    risk   = entry - sl
    reward = tp - entry
    if risk <= 0:
        return None
    rr = reward / risk
    if rr < MIN_RR:
        return None

    return {
        "setup":            "HH Breakout Retest",
        "entry":            entry,
        "tp":               tp,
        "sl":               sl,
        "rr":               rr,
        "rsi":              rsi_val,
        "pullback_candles": 0,
        "pullback_low":     sl,
        "ema_touched":      "Broken Level",
        "ema20":            sig_candle["ema20"],
        "ema50":            ema50_val,
        "vol_mult":         sig_candle["vol"] / vol_avg,
        "df":               df,
        "broken_level":     swing_high_val,
    }


# ================= CHART =================
def make_chart(sig, symbol, tf):
    df  = sig["df"].tail(60).reset_index(drop=True)
    n   = len(df)
    cw  = 0.4

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(10, 5), dpi=180,
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    axv.set_facecolor("#0d0d0d")

    price_range = max(df["high"].max() - df["low"].min(), 1e-9)

    for i, row in df.iterrows():
        col = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.plot([i, i], [row["low"], row["high"]], color=col, linewidth=0.8, zorder=1)
        h = max(abs(row["close"] - row["open"]), price_range * 0.001)
        ax.add_patch(plt.Rectangle(
            (i - cw / 2, min(row["open"], row["close"])),
            cw, h, color=col, zorder=2
        ))

    ax.plot(df.index, df["ema20"], color="#2196F3", linewidth=1.0, zorder=3)
    ax.plot(df.index, df["ema50"], color="#FF9800", linewidth=1.0, zorder=3)

    # Highlight signal candle
    sig_i = n - 2
    ax.add_patch(plt.Rectangle(
        (sig_i - cw / 2 - 0.1, df.iloc[sig_i]["low"]),
        cw + 0.2,
        df.iloc[sig_i]["high"] - df.iloc[sig_i]["low"],
        color="#FFD700", alpha=0.15, zorder=0
    ))

    # EMA Pullback: shade pullback zone
    if sig["setup"] == "EMA Pullback" and sig["pullback_candles"] > 0:
        pb_start = max(0, sig_i - sig["pullback_candles"] - 1)
        ax.axvspan(pb_start, sig_i - 0.5, color="#ef5350", alpha=0.06, zorder=0)

    # HH Retest: draw broken level
    if sig["setup"] == "HH Breakout Retest" and sig["broken_level"]:
        ax.axhline(sig["broken_level"], color="#9C27B0",
                   linewidth=1.0, linestyle="--", alpha=0.8)
        ax.text(2, sig["broken_level"], " Broken Level",
                color="#9C27B0", fontsize=5, va="bottom", alpha=0.9)

    # TP / SL / Entry
    ax.axhline(sig["entry"], color="#FFD700", linewidth=0.9, linestyle="-",  alpha=0.9)
    ax.axhline(sig["tp"],    color="#00e676", linewidth=0.9, linestyle="-",  alpha=0.9)
    ax.axhline(sig["sl"],    color="#ff1744", linewidth=0.9, linestyle="--", alpha=0.8)

    ax.text(n + 0.3, sig["tp"],    f"TP {fmt_price(sig['tp'])}",
            color="#00e676", fontsize=5, va="center")
    ax.text(n + 0.3, sig["entry"], f"E  {fmt_price(sig['entry'])}",
            color="#FFD700", fontsize=5, va="center")
    ax.text(n + 0.3, sig["sl"],    f"SL {fmt_price(sig['sl'])}",
            color="#ff1744", fontsize=5, va="center")

    # Volume
    max_vol = df["vol"].max() or 1
    scaled  = df["vol"] / max_vol * 0.1 * price_range
    v_cols  = ["#26a69a" if df.loc[i,"close"] >= df.loc[i,"open"]
               else "#ef5350" for i in df.index]
    axv.bar(df.index, scaled, color=v_cols, width=cw)
    axv.bar([sig_i], [scaled.iloc[sig_i]], color="#FFD700", width=cw)

    for a in [ax, axv]:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    legend_els = [
        mpatches.Patch(color="#2196F3", label="EMA20"),
        mpatches.Patch(color="#FF9800", label="EMA50"),
        mpatches.Patch(color="#FFD700", alpha=0.5, label="Signal"),
    ]
    if sig["setup"] == "HH Breakout Retest":
        legend_els.append(mpatches.Patch(color="#9C27B0", label="Broken Level"))

    ax.legend(handles=legend_els, loc="upper left", fontsize=5,
              facecolor="#1a1a1a", edgecolor="none", labelcolor="white",
              framealpha=0.8, ncol=4)

    ax.set_title(f"{clean_sym(symbol)}  {tf}  |  {sig['setup']}",
                 color="white", fontsize=7, pad=3, loc="right")

    buf = io.BytesIO()
    plt.tight_layout(pad=0.3)
    plt.savefig(buf, format="png", facecolor="#0d0d0d")
    plt.close(fig)
    buf.seek(0)
    return buf


# ================= FORMAT MESSAGE =================
def format_message(sig, symbol, tf, htf):
    tp_pct   = (sig["tp"]    - sig["entry"]) / sig["entry"] * 100
    sl_pct   = (sig["entry"] - sig["sl"])    / sig["entry"] * 100
    tf_emoji = {"15m": "⚡", "30m": "🕐", "1h": "💎"}.get(tf, "📊")
    size_tip = {"15m": "small", "30m": "medium", "1h": "full"}.get(tf, "medium")

    if sig["setup"] == "EMA Pullback":
        setup_line = (
            f"📍 *EMA Pullback — Uptrend*\n"
            f"*Touched:* {sig['ema_touched']}\n"
            f"*Pullback:* {sig['pullback_candles']} red candles\n"
            f"*HTF ({htf}):* Bullish ✅"
        )
    else:
        setup_line = (
            f"🔁 *Higher High Breakout Retest*\n"
            f"*Broken level:* `{fmt_price(sig['broken_level'])}`\n"
            f"*HTF ({htf}):* Bullish ✅"
        )

    return (
        f"{tf_emoji} *{clean_sym(symbol)}* — {tf}\n\n"
        f"{setup_line}\n\n"
        f"🟢 *LONG*\n\n"
        f"🎯 *Entry:* `{fmt_price(sig['entry'])}`\n"
        f"✅ *TP:* `{fmt_price(sig['tp'])}` _(+{tp_pct:.1f}%)_\n"
        f"🛑 *SL:* `{fmt_price(sig['sl'])}` _(-{sl_pct:.1f}%)_\n"
        f"⚖️ *R:R:* `{sig['rr']:.1f}:1`\n\n"
        f"📊 *RSI:* `{sig['rsi']:.1f}`\n"
        f"📦 *Vol:* `{sig['vol_mult']:.1f}x` avg\n\n"
        f"💡 _Size tip: {size_tip} position_"
    )


# ================= MAIN SCANNER =================
async def run_scan(context: ContextTypes.DEFAULT_TYPE):
    log("🔍 Scan started")
    subscribers = read_subscribers()
    if not subscribers:
        return

    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        log(f"Ticker fetch failed: {e}")
        return

    futures = [
        t for s, t in tickers.items()
        if "/USDT" in s and ":USDT" in s and t.get("percentage") is not None
    ]
    gainers = sorted(futures, key=lambda x: x["percentage"], reverse=True)[:TOP_MOVERS_COUNT]
    losers  = sorted(futures, key=lambda x: x["percentage"])[:TOP_MOVERS_COUNT]
    targets = list({t["symbol"]: t for t in gainers + losers}.values())

    signals_fired = 0

    for t in targets:
        symbol = t["symbol"]

        for tf in TIMEFRAMES:
            # HTF trend check — uses cache, only fetches when cache expires
            htf         = HTF_REQUIRED[tf]
            htf_bullish = get_htf_trend(symbol, htf)

            # Fetch OHLCV with rate limit retry built in
            df = fetch_ohlcv_safe(symbol, tf, limit=120)
            if df is None or len(df) < 60:
                continue

            # Run both detectors
            candidates = []

            ema_sig = detect_ema_pullback(df, tf, htf_bullish)
            if ema_sig:
                candidates.append(("ema_pullback", ema_sig))

            hh_sig = detect_hh_retest(df, tf, htf_bullish)
            if hh_sig:
                candidates.append(("hh_retest", hh_sig))

            for sig_type, sig in candidates:
                cd  = SIGNAL_COOLDOWN.get(tf, 900) if sig_type == "ema_pullback" \
                      else HH_COOLDOWN.get(tf, 3600)
                key = f"{symbol}_{tf}_{sig_type}"

                if not cooldown_ok(key, cd):
                    continue

                mark_cooldown(key)
                log_signal(symbol, tf, sig["setup"], sig["entry"],
                           sig["tp"], sig["sl"], sig["rr"], sig["rsi"])
                signals_fired += 1

                msg   = format_message(sig, symbol, tf, htf)
                chart = make_chart(sig, symbol, tf)

                log(f"🚨 {sig['setup']} | {clean_sym(symbol)} {tf} | "
                    f"R:R {sig['rr']:.1f} | RSI {sig['rsi']:.1f} | HTF {htf} ✅")

                for chat in subscribers:
                    try:
                        await context.bot.send_photo(
                            chat_id=chat,
                            photo=chart,
                            caption=msg,
                            parse_mode="Markdown"
                        )
                        chart.seek(0)
                        await asyncio.sleep(1.5)
                    except Exception as e:
                        log(f"Send error {chat}: {e}")

    log(f"✅ Scan done — {signals_fired} signal(s) fired")


# ================= BACKGROUND LOOP =================
async def scan_loop(context: ContextTypes.DEFAULT_TYPE):
    global _scan_running
    if _scan_running:
        return
    _scan_running = True
    log("🤖 Scan loop started")
    while True:
        try:
            await run_scan(context)
        except Exception as e:
            log(f"Loop error: {e}")
        await asyncio.sleep(SCAN_INTERVAL)

async def post_init(app):
    # Schedule the scan loop as a repeating job using job_queue
    # This is more reliable than asyncio.create_task on hosted environments
    app.job_queue.run_repeating(
        scan_job,
        interval=SCAN_INTERVAL,
        first=10  # start 10 seconds after bot launches
    )
    log("✅ Scan job scheduled")

async def scan_job(context: ContextTypes.DEFAULT_TYPE):
    """Wrapper for job_queue — catches all errors so loop never dies."""
    try:
        await run_scan(context)
    except Exception as e:
        log(f"Scan job error: {e}")


# ================= TELEGRAM COMMANDS =================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return
    save_subscriber(str(update.effective_chat.id))
    await safe_reply(update,
        "✅ *EMA Pullback + HH Retest Bot*\n\n"
        "Subscribed to signals.\n\n"
        "*Setups:*\n"
        "⚡ EMA Pullback — dip to EMA in uptrend\n"
        "🔁 HH Breakout Retest — bounce off broken resistance\n\n"
        "*Timeframes:* 15m · 30m · 1h\n\n"
        "*Every signal requires:*\n"
        "• Higher TF bullish ✅\n"
        "• Clean signal candle (no big upper wick) ✅\n"
        "• RSI 38–65 ✅\n"
        "• Volume above average ✅\n"
        "• R:R ≥ 1.8 ✅\n\n"
        "*HTF alignment:*\n"
        "15m → 1h must be bullish\n"
        "30m → 4h must be bullish\n"
        "1h → 1d must be bullish\n\n"
        "/scan /stats /help /stop",
        parse_mode="Markdown"
    )

async def safe_reply(update: Update, text: str, **kwargs):
    """Safely reply in both private chats and groups."""
    try:
        if update.message:
            await update.message.reply_text(text, **kwargs)
        elif update.effective_chat:
            await update.effective_chat.send_message(text, **kwargs)
    except Exception as e:
        log(f"Reply error: {e}")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return
    remove_subscriber(str(update.effective_chat.id))
    await safe_reply(update, "❌ Unsubscribed. /start to resubscribe.")

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return
    cid = str(update.effective_chat.id)
    key = f"manual_{cid}"
    if key in ALERT_MEMORY and time.time() - ALERT_MEMORY[key] < 120:
        await safe_reply(update, "⏱ Wait 2 min between manual scans.")
        return
    ALERT_MEMORY[key] = time.time()
    await safe_reply(update, "⚡ Scanning all timeframes...")
    await run_scan(context)
    await safe_reply(update, "✅ Done!")

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return
    today      = datetime.now().strftime("%Y-%m-%d")
    today_sigs = []
    if os.path.exists(SIGNAL_LOG_FILE):
        with open(SIGNAL_LOG_FILE) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("timestamp", "").startswith(today):
                    today_sigs.append(row)

    ema_count = sum(1 for s in today_sigs if "EMA"  in s.get("setup", ""))
    hh_count  = sum(1 for s in today_sigs if "HH"   in s.get("setup", ""))
    by_tf     = {"15m": 0, "30m": 0, "1h": 0}
    for s in today_sigs:
        tf = s.get("tf", "")
        if tf in by_tf:
            by_tf[tf] += 1

    total_all = 0
    if os.path.exists(SIGNAL_LOG_FILE):
        with open(SIGNAL_LOG_FILE) as f:
            total_all = sum(1 for _ in f) - 1

    await safe_reply(update,
        f"📊 *Stats — {today}*\n\n"
        f"*By setup:*\n"
        f"⚡ EMA Pullback: `{ema_count}`\n"
        f"🔁 HH Retest: `{hh_count}`\n\n"
        f"*By timeframe:*\n"
        f"15m: `{by_tf['15m']}` · 30m: `{by_tf['30m']}` · 1h: `{by_tf['1h']}`\n\n"
        f"Today total: `{len(today_sigs)}`\n"
        f"All-time: `{total_all}`",
        parse_mode="Markdown"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return
    await safe_reply(update,
        "📖 *How to trade signals*\n\n"
        "*EMA Pullback:*\n"
        "Price dipped to EMA in uptrend and bounced.\n"
        "Entry at close, SL below pullback low, TP 2–5%.\n\n"
        "*HH Breakout Retest:*\n"
        "Price broke above a swing high, pulled back to retest it.\n"
        "Broken resistance becomes support. Entry at the bounce.\n\n"
        "*Both require HTF to be bullish — this is the most\n"
        "important filter. If 1h/4h/1d is bearish, no signal fires.*\n\n"
        "*Skip signal if:*\n"
        "• BTC is dumping hard right now\n"
        "• News event in next 30 min\n"
        "• You don't recognize the coin\n\n"
        "/start /stop /scan /stats /help",
        parse_mode="Markdown"
    )


# ================= MAIN =================
def main():
    log("🚀 Bot starting...")
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop",  cmd_stop))
    app.add_handler(CommandHandler("scan",  cmd_scan))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
