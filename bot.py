"""
Automatic PAPER trading bot - Bollinger Band M / W patterns.
Runs on GitHub Actions every 15 minutes. No real money. No broker.

Strategy
- Bollinger Bands (20, 2).
- W bottom (long): 1st low at/below lower band, 2nd HIGHER low inside the band,
  entry when price closes above the neckline.
- M top (short): 1st high at/above upper band, 2nd LOWER high inside the band,
  entry when price closes below the neckline. (Shorts only for US stocks + crypto.)
- Stop-loss just beyond the 2nd low/high (+ 0.25 ATR buffer).
- Target = measured move (pattern height projected from neckline),
  capped at the last major swing if that is closer.
- Trade only if reward : risk >= 2.
- Risk per trade 2.5% of equity. Max total open risk 10%. Stop new trades if
  equity falls 15% from start. Move stop to breakeven at +1R. Time exit 20 bars.
"""
import json
import math
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

import numpy as np
import pandas as pd

# ----------------------------- SETTINGS ------------------------------------
START_EQUITY = 100_000.0        # USD paper money
RISK_PER_TRADE = 0.025          # 2.5% of equity
MAX_OPEN_RISK = 0.10            # 10% of equity at risk across all trades
MAX_NOTIONAL_PER_TRADE = 0.25   # one trade can't be more than 25% of equity
MAX_TOTAL_NOTIONAL = 1.00       # no leverage
STOP_NEW_TRADES_DRAWDOWN = 0.15 # stop opening trades if equity down 15%
MIN_RR = 2.0
RUN_DAYS = 90
TIME_EXIT_BARS = 20
SLIPPAGE = 0.0005               # 0.05% on entries and stop/time exits
BB_LEN, BB_STD, ATR_LEN = 20, 2.0, 14
PIVOT_K = 3
EMAIL_HOUR_IST = 22             # daily email after 10 PM India time
STATE_FILE = "state.json"
TRADES_FILE = "trades.csv"

NIFTY = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
         "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT",
         "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "NESTLEIND", "ULTRACEMCO", "WIPRO"]
US = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "NFLX"]
CRYPTO = ["BTC-USD"]

INSTRUMENTS = {}
for s in NIFTY:
    INSTRUMENTS[s + ".NS"] = {"market": "NSE", "short": False}
for s in US:
    INSTRUMENTS[s] = {"market": "US", "short": True}
for s in CRYPTO:
    INSTRUMENTS[s] = {"market": "CRYPTO", "short": True}

TIMEFRAMES = {"1d": {"period": "1y", "delta": timedelta(days=1)},
              "15m": {"period": "30d", "delta": timedelta(minutes=15)}}

NOW = datetime.now(timezone.utc)
IST = timezone(timedelta(hours=5, minutes=30))


# ----------------------------- HELPERS -------------------------------------
def log(*a):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "start_date": NOW.isoformat(),
        "end_date": (NOW + timedelta(days=RUN_DAYS)).isoformat(),
        "realized": 0.0,
        "positions": [],
        "signals_seen": [],
        "closed_count": 0,
        "usd_per_inr": 1 / 88.0,
        "last_email_date": "",
        "final_sent": False,
    }


def save_state(st):
    st["signals_seen"] = st["signals_seen"][-3000:]
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=1)


def market_open(market, now=NOW):
    if market == "CRYPTO":
        return True
    if now.weekday() >= 5:
        return False
    if market == "NSE":
        t = now.astimezone(IST)
        return (9, 10) <= (t.hour, t.minute) <= (15, 45)
    # US: 13:25-20:15 UTC covers 9:30-16:00 New York in both summer and winter time
    m = now.hour * 60 + now.minute
    return 13 * 60 + 25 <= m <= 21 * 60 + 15


def fetch(symbol, tf):
    import yfinance as yf
    cfg = TIMEFRAMES[tf]
    df = yf.Ticker(symbol).history(period=cfg["period"], interval=tf, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close"]].dropna().copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = ["open", "high", "low", "close"]
    return df


def closed_only(df, tf):
    delta = TIMEFRAMES[tf]["delta"]
    return df[df.index + delta <= pd.Timestamp(NOW)]


def add_indicators(df):
    df = df.copy()
    ma = df["close"].rolling(BB_LEN).mean()
    sd = df["close"].rolling(BB_LEN).std(ddof=0)
    df["mid"], df["upper"], df["lower"] = ma, ma + BB_STD * sd, ma - BB_STD * sd
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()
    return df


def pivots(arr, k, kind):
    """Indices of confirmed pivot lows ('low') or highs ('high'). Ties allowed (first one counts)."""
    out = []
    n = len(arr)
    for i in range(k, n - k):
        left, right = arr[i - k:i], arr[i + 1:i + k + 1]
        if kind == "low" and left.min() > arr[i] and right.min() >= arr[i]:
            out.append(i)
        if kind == "high" and left.max() < arr[i] and right.max() <= arr[i]:
            out.append(i)
    return out


# ----------------------------- PATTERNS ------------------------------------
def find_signal(df_closed, latest_price, allow_short):
    """Return a signal dict or None. df_closed has indicators, closed bars only."""
    df = df_closed.dropna(subset=["lower", "atr"]).reset_index()
    n = len(df)
    if n < 60:
        return None
    tcol = df.columns[0]
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    up, lw, atr = df["upper"].values, df["lower"].values, df["atr"].values
    last = n - 1

    # ---- W bottom (long) ----
    pl = [i for i in pivots(lo, PIVOT_K, "low") if i >= n - 45]
    if len(pl) >= 2:
        b = pl[-1]
        for a in reversed(pl[:-1]):
            if not (5 <= b - a <= 35):
                continue
            if not (lo[a] <= lw[a] and lo[b] > lo[a] and lo[b] > lw[b]):
                continue
            neck = hi[a:b + 1].max()
            if neck - lo[a] < 1.5 * atr[b]:
                continue
            if not (1 <= last - b <= 12):
                continue
            if not (cl[last] > neck and (cl[b + 1:last] <= neck).all()):
                continue
            entry = float(latest_price)
            if entry > neck + 0.5 * atr[last]:
                continue  # already ran away
            stop = lo[b] - 0.25 * atr[b]
            target = neck + (neck - lo[a])
            swing = hi[max(0, a - 60):a + 1].max()
            if entry < swing < target:
                target = swing
            return _finish("LONG", "W", df[tcol].iloc[b], entry, stop, target)

    # ---- M top (short) ----
    if allow_short:
        ph = [i for i in pivots(hi, PIVOT_K, "high") if i >= n - 45]
        if len(ph) >= 2:
            b = ph[-1]
            for a in reversed(ph[:-1]):
                if not (5 <= b - a <= 35):
                    continue
                if not (hi[a] >= up[a] and hi[b] < hi[a] and hi[b] < up[b]):
                    continue
                neck = lo[a:b + 1].min()
                if hi[a] - neck < 1.5 * atr[b]:
                    continue
                if not (1 <= last - b <= 12):
                    continue
                if not (cl[last] < neck and (cl[b + 1:last] >= neck).all()):
                    continue
                entry = float(latest_price)
                if entry < neck - 0.5 * atr[last]:
                    continue
                stop = hi[b] + 0.25 * atr[b]
                target = neck - (hi[a] - neck)
                swing = lo[max(0, a - 60):a + 1].min()
                if target < swing < entry:
                    target = swing
                return _finish("SHORT", "M", df[tcol].iloc[b], entry, stop, target)
    return None


def _finish(side, pattern, pivot_time, entry, stop, target):
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0 or risk / entry > 0.10 or risk / entry < 0.002:
        return None
    if side == "LONG" and not (stop < entry < target):
        return None
    if side == "SHORT" and not (target < entry < stop):
        return None
    if reward / risk < MIN_RR:
        return None
    return {"side": side, "pattern": pattern, "key_time": str(pivot_time),
            "entry": entry, "stop": float(stop), "target": float(target), "rr": reward / risk}


# ----------------------------- ACCOUNT -------------------------------------
def fx_of(symbol, st):
    return st["usd_per_inr"] if INSTRUMENTS[symbol]["market"] == "NSE" else 1.0


def unrealized(st):
    total = 0.0
    for p in st["positions"]:
        d = 1 if p["side"] == "LONG" else -1
        total += (p["last_price"] - p["entry"]) * p["qty"] * d * fx_of(p["symbol"], st)
    return total


def equity(st):
    return START_EQUITY + st["realized"] + unrealized(st)


def log_trade(row):
    new = not os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a") as f:
        if new:
            f.write("exit_time,symbol,tf,side,pattern,entry_time,entry,exit,qty,pnl_usd,reason\n")
        f.write(",".join(str(row[k]) for k in ["exit_time", "symbol", "tf", "side", "pattern", "entry_time",
                                                 "entry", "exit", "qty", "pnl_usd", "reason"]) + "\n")


def close_position(st, p, price, reason, when):
    d = 1 if p["side"] == "LONG" else -1
    if reason in ("STOP", "TIME", "BREAKEVEN"):
        price = price * (1 - SLIPPAGE * d)
    pnl = (price - p["entry"]) * p["qty"] * d * fx_of(p["symbol"], st)
    st["realized"] += pnl
    st["closed_count"] += 1
    st["positions"] = [x for x in st["positions"] if x is not p]
    log_trade({"exit_time": when, "symbol": p["symbol"], "tf": p["tf"], "side": p["side"],
               "pattern": p["pattern"], "entry_time": p["entry_time"], "entry": round(p["entry"], 4),
               "exit": round(price, 4), "qty": p["qty"], "pnl_usd": round(pnl, 2), "reason": reason})
    log(f"CLOSED {p['symbol']} {p['tf']} {p['side']} {reason} pnl={pnl:.2f}")


def manage_positions(st, cache):
    for p in list(st["positions"]):
        try:
            df = cache(p["symbol"], p["tf"])
            if df is None:
                continue
            p["last_price"] = float(df["close"].iloc[-1])
            delta = TIMEFRAMES[p["tf"]]["delta"]
            long_ = p["side"] == "LONG"
            one_r = abs(p["entry"] - p["init_stop"])
            after = pd.Timestamp(p.get("last_ts") or p["entry_time"])
            new = df[df.index > after]
            done = False
            for ts, bar in new.iterrows():
                is_closed = ts + delta <= pd.Timestamp(NOW)
                hit_stop = bar["low"] <= p["stop"] if long_ else bar["high"] >= p["stop"]
                hit_tgt = bar["high"] >= p["target"] if long_ else bar["low"] <= p["target"]
                if hit_stop:  # if stop and target are both inside one bar, assume the worst
                    be = abs(p["stop"] - p["entry"]) < 1e-9
                    close_position(st, p, p["stop"], "BREAKEVEN" if be else "STOP", str(ts))
                    done = True
                    break
                if hit_tgt:
                    close_position(st, p, p["target"], "TARGET", str(ts))
                    done = True
                    break
                if not is_closed:
                    continue  # forming bar: only stop/target checked, no state changes
                reached = bar["high"] >= p["entry"] + one_r if long_ else bar["low"] <= p["entry"] - one_r
                if reached:
                    p["stop"] = p["entry"]
                p["bars_held"] = p.get("bars_held", 0) + 1
                p["last_ts"] = ts.isoformat()
                if p["bars_held"] >= TIME_EXIT_BARS:
                    close_position(st, p, float(bar["close"]), "TIME", str(ts))
                    done = True
                    break
        except Exception as e:  # never crash the whole run
            log("manage error", p["symbol"], e)


def open_open_risk(st):
    return sum(p["risk_usd"] for p in st["positions"] if abs(p["stop"] - p["entry"]) > 1e-9)


def try_enter(st, symbol, tf, sig):
    eq = equity(st)
    if eq <= START_EQUITY * (1 - STOP_NEW_TRADES_DRAWDOWN):
        return False
    if any(p["symbol"] == symbol for p in st["positions"]):
        return False
    fx = fx_of(symbol, st)
    side = sig["side"]
    d = 1 if side == "LONG" else -1
    entry = sig["entry"] * (1 + SLIPPAGE * d)
    risk_unit = abs(entry - sig["stop"])
    if risk_unit <= 0:
        return False
    risk_usd = eq * RISK_PER_TRADE
    qty = risk_usd / (risk_unit * fx)
    max_notional = eq * MAX_NOTIONAL_PER_TRADE
    qty = min(qty, max_notional / (entry * fx))
    cur_notional = sum(p["qty"] * p["entry"] * fx_of(p["symbol"], st) for p in st["positions"])
    qty = min(qty, (eq * MAX_TOTAL_NOTIONAL - cur_notional) / (entry * fx))
    qty = math.floor(qty) if INSTRUMENTS[symbol]["market"] != "CRYPTO" else math.floor(qty * 1e5) / 1e5
    if qty <= 0:
        return False
    real_risk = qty * risk_unit * fx
    if open_open_risk(st) + real_risk > eq * MAX_OPEN_RISK:
        return False
    st["positions"].append({
        "symbol": symbol, "tf": tf, "side": side, "pattern": sig["pattern"],
        "entry_time": NOW.isoformat(), "entry": entry, "stop": sig["stop"], "init_stop": sig["stop"],
        "target": sig["target"], "qty": qty, "risk_usd": real_risk, "rr": round(sig["rr"], 2),
        "last_price": entry, "bars_held": 0,
    })
    log(f"OPENED {symbol} {tf} {side} {sig['pattern']} qty={qty} entry={entry:.4f} "
        f"stop={sig['stop']:.4f} target={sig['target']:.4f} rr={sig['rr']:.2f}")
    return True


# ----------------------------- EMAIL ---------------------------------------
def send_email(subject, body):
    user, pw = os.environ.get("GMAIL_ADDRESS"), os.environ.get("GMAIL_APP_PASSWORD")
    if not user or not pw:
        log("Email secrets missing - skipping email")
        return False
    to = os.environ.get("EMAIL_TO", user)
    msg = MIMEText(body)
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
            s.login(user, pw)
            s.sendmail(user, [to], msg.as_string())
        log("Email sent")
        return True
    except Exception as e:
        log("Email failed:", e)
        return False


def report(st, final=False):
    eq = equity(st)
    day = (NOW - pd.Timestamp(st["start_date"]).to_pydatetime()).days
    lines = [f"PAPER TRADING BOT - {'FINAL REPORT' if final else 'daily update'}",
             f"Day {min(day, RUN_DAYS)} of {RUN_DAYS}", "",
             f"Equity:       ${eq:,.2f}  ({(eq / START_EQUITY - 1) * 100:+.2f}% from $100,000)",
             f"Realized P&L: ${st['realized']:,.2f}",
             f"Unrealized:   ${unrealized(st):,.2f}",
             f"Trades closed: {st['closed_count']}", ""]
    if os.path.exists(TRADES_FILE):
        t = pd.read_csv(TRADES_FILE)
        if len(t):
            wins = (t["pnl_usd"] > 0).sum()
            lines.append(f"Win rate: {wins / len(t) * 100:.0f}%  |  Avg win ${t[t.pnl_usd > 0].pnl_usd.mean() if wins else 0:,.0f}"
                         f"  |  Avg loss ${t[t.pnl_usd <= 0].pnl_usd.mean() if wins < len(t) else 0:,.0f}")
            lines.append("")
            lines.append("Last 5 closed trades:")
            for _, r in t.tail(5).iterrows():
                lines.append(f"  {r.symbol} {r.tf} {r.side} {r.reason}: ${r.pnl_usd:,.2f}")
            lines.append("")
    lines.append(f"Open trades: {len(st['positions'])}")
    for p in st["positions"]:
        d = 1 if p["side"] == "LONG" else -1
        pnl = (p["last_price"] - p["entry"]) * p["qty"] * d * fx_of(p["symbol"], st)
        lines.append(f"  {p['symbol']} {p['tf']} {p['side']} {p['pattern']} entry {p['entry']:.2f} "
                     f"stop {p['stop']:.2f} target {p['target']:.2f}  now ${pnl:,.0f}")
    if eq <= START_EQUITY * (1 - STOP_NEW_TRADES_DRAWDOWN):
        lines += ["", "NOTE: account is down 15%+, bot has paused new trades (safety rule)."]
    lines += ["", "This is paper trading with fake money. Real trading results can differ."]
    return "\n".join(lines)


# ----------------------------- MAIN ----------------------------------------
def main():
    if os.environ.get("TEST_EMAIL") == "1":
        ok = send_email("Paper bot: test email", "Your paper trading bot can send you email. Setup is correct.")
        sys.exit(0 if ok else 1)

    st = load_state()
    ended = NOW >= pd.Timestamp(st["end_date"]).to_pydatetime()
    cache_store = {}

    def cache(symbol, tf):
        k = (symbol, tf)
        if k not in cache_store:
            try:
                cache_store[k] = fetch(symbol, tf)
            except Exception as e:
                log("fetch error", symbol, tf, e)
                cache_store[k] = None
        return cache_store[k]

    # USD/INR for Indian stocks
    try:
        import yfinance as yf
        fx = yf.Ticker("USDINR=X").history(period="5d", interval="1d")["Close"].dropna().iloc[-1]
        if 60 < fx < 150:
            st["usd_per_inr"] = float(1 / fx)
    except Exception as e:
        log("fx error", e)

    manage_positions(st, cache)

    if ended:
        # close everything at last price and send final report once
        for p in list(st["positions"]):
            close_position(st, p, p["last_price"], "END", NOW.isoformat())
        if not st["final_sent"]:
            if send_email("Paper bot: FINAL 90-day report", report(st, final=True)):
                st["final_sent"] = True
        save_state(st)
        return

    scan_daily = NOW.minute < 15  # daily-candle scan once per hour is plenty
    for symbol, meta in INSTRUMENTS.items():
        for tf in ("1d", "15m"):
            if tf == "1d" and not scan_daily:
                continue
            if tf == "15m" and not market_open(meta["market"]):
                continue
            try:
                df = cache(symbol, tf)
                if df is None or len(df) < 80:
                    continue
                latest = float(df["close"].iloc[-1])
                dfc = add_indicators(closed_only(df, tf))
                sig = find_signal(dfc, latest, meta["short"])
                if not sig:
                    continue
                key = f"{symbol}|{tf}|{sig['pattern']}|{sig['key_time']}"
                if key in st["signals_seen"]:
                    continue
                st["signals_seen"].append(key)
                try_enter(st, symbol, tf, sig)
            except Exception as e:
                log("scan error", symbol, tf, e)

    # daily email after 10 PM IST, once per day
    ist = NOW.astimezone(IST)
    today = ist.strftime("%Y-%m-%d")
    if ist.hour >= EMAIL_HOUR_IST and st["last_email_date"] != today:
        if send_email(f"Paper bot daily update - {today}", report(st)):
            st["last_email_date"] = today

    save_state(st)
    log(f"Done. Equity ${equity(st):,.2f}, open trades {len(st['positions'])}")


if __name__ == "__main__":
    main()
