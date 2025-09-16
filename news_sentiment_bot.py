#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News/Sentiment Trading-Bot (Stocks) – Lightweight MVP
- Preis-Event-getrieben: Erst bei 5-Minuten-Spike werden News für das Symbol geholt (yfinance).
- Sentiment per VADER (leichtgewichtig, keine großen Modelle nötig).
- Trading via Alpaca (alpaca-py), mit Bracket-Order (TP/SL).
- De-duplication: Jede Headline wird nur einmal gehandelt; Cooldown je Symbol.
- Optional nur während Markt offen; optional kein Short (Paper: true/false).

Späteres Upgrade möglich:
- FinBERT (HF) statt VADER
- Polygon/Alpaca News-API statt yfinance
- Dashboard/Logs/DB etc.
"""

import os
import time
import json
import math
import argparse
import datetime as dt
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import pytz

# News
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Alpaca
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# ======================
# Konfiguration aus ENV
# ======================
API_KEY = os.getenv("APCA_API_KEY_ID", "")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
DATA_FEED = os.getenv("APCA_API_DATA_FEED", "iex").lower()

WATCHLIST = [s.strip().upper() for s in os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,TSLA,AMZN").split(",") if s.strip()]

RISK_USD_PER_TRADE = float(os.getenv("RISK_USD_PER_TRADE", "200"))
TP_PCT = float(os.getenv("TP_PCT", "0.02"))      # 2%
SL_PCT = float(os.getenv("SL_PCT", "0.01"))      # 1%
PRICE_SPIKE_PCT_5M = float(os.getenv("PRICE_SPIKE_PCT_5M", "2.0"))  # 5-Min Änderung in %
NEWS_POS_TH = float(os.getenv("NEWS_POS_TH", "0.40"))
NEWS_NEG_TH = float(os.getenv("NEWS_NEG_TH", "-0.40"))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "30"))
POLLING_SECONDS = int(os.getenv("POLLING_SECONDS", "60"))
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false").lower() == "true"
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() == "true"

STATE_FILE = "news_state.json"  # gemerkte Headlines & Symbol-Cooldowns

# ======================
# Helpers
# ======================
UTC = pytz.UTC
analyzer = SentimentIntensityAnalyzer()

def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=UTC)

def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return {"seen_ids": [], "symbol_cooldown": {}}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"seen_ids": [], "symbol_cooldown": {}}

def save_state(state: Dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception:
        pass

def is_market_open(trading: TradingClient) -> bool:
    try:
        clock = trading.get_clock()
        return bool(clock.is_open)
    except Exception:
        # Fallback: lieber handeln zulassen als komplett blocken
        return True

def build_clients() -> Tuple[TradingClient, StockHistoricalDataClient]:
    if not (API_KEY and API_SECRET):
        raise RuntimeError("❌ Alpaca API Keys fehlen (APCA_API_KEY_ID / APCA_API_SECRET_KEY).")
    paper = "paper" in API_BASE_URL
    trading = TradingClient(API_KEY, API_SECRET, paper=paper)
    market = StockHistoricalDataClient(API_KEY, API_SECRET)
    return trading, market

def fetch_last_minute_prices(market: StockHistoricalDataClient, symbol: str, minutes: int = 6) -> Optional[pd.DataFrame]:
    """
    Holt die letzten ~6 Minuten Bars, um die 5-Minuten-Änderung zu messen.
    """
    end = now_utc()
    start = end - dt.timedelta(minutes=minutes + 2)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
        adjustment="raw",
        feed=DATA_FEED,
        limit=minutes + 5,
    )
    try:
        resp = market.get_stock_bars(req)
        bars = resp.data.get(symbol, [])
        if not bars:
            return None
        rows = []
        for b in bars:
            rows.append({
                "t": pd.Timestamp(b.timestamp).tz_convert("UTC"),
                "c": float(b.close),
            })
        df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[WARN] {symbol}: Fehler beim Holen von Minute-Bars: {e}")
        return None

def compute_5m_change_pct(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 6:
        return None
    last = df["c"].iloc[-1]
    prev5 = df["c"].iloc[-6]  # ca. 5 Min vor letzter Bar
    if prev5 <= 0:
        return None
    return (last - prev5) / prev5 * 100.0

def fetch_news_for_symbol(symbol: str, max_items: int = 6) -> List[Dict]:
    """
    Holt News via yfinance (kostenlos). Struktur variiert je nach Quelle.
    Wir extrahieren: id (oder title+time), title, published (epoch), link.
    """
    try:
        t = yf.Ticker(symbol)
        items = t.news or []
    except Exception as e:
        print(f"[WARN] {symbol}: News-Load-Fehler: {e}")
        items = []

    out = []
    for it in items[:max_items]:
        title = it.get("title") or ""
        link = it.get("link") or ""
        provider_ts = it.get("providerPublishTime")  # epoch sek
        published = int(provider_ts) if provider_ts else 0
        # id bauen: title + published
        nid = f"{symbol}:{published}:{title[:80]}"
        out.append({
            "id": nid,
            "symbol": symbol,
            "title": title,
            "published": published,
            "link": link
        })
    return out

def sentiment_score(text: str) -> float:
    """
    VADER compound Score in [-1, +1]
    """
    s = analyzer.polarity_scores(text)
    return float(s.get("compound", 0.0))

def eligible_by_cooldown(state: Dict, symbol: str) -> bool:
    cd_map = state.get("symbol_cooldown", {})
    until_ts = cd_map.get(symbol)
    if not until_ts:
        return True
    return now_utc().timestamp() > float(until_ts)

def set_cooldown(state: Dict, symbol: str, minutes: int) -> None:
    cd_map = state.setdefault("symbol_cooldown", {})
    cd_map[symbol] = now_utc().timestamp() + minutes * 60.0

def already_seen(state: Dict, nid: str) -> bool:
    return nid in set(state.get("seen_ids", []))

def mark_seen(state: Dict, nid: str) -> None:
    ids: List[str] = state.setdefault("seen_ids", [])
    ids.append(nid)
    # begrenzen (Speicher)
    if len(ids) > 5000:
        del ids[:1000]

def get_last_price_from_df(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    return float(df["c"].iloc[-1])

def place_bracket_order(
    trading: TradingClient,
    symbol: str,
    side: OrderSide,
    notional_usd: float,
    last_price: float,
    tp_pct: float,
    sl_pct: float,
    extended: bool,
) -> None:
    """
    Bracket-Order: Market-Einstieg mit TP/SL um last_price herum.
    """
    if last_price is None or last_price <= 0:
        raise ValueError("Kein valider last_price für Bracket.")

    if side == OrderSide.BUY:
        tp_limit = round(last_price * (1.0 + tp_pct), 2)
        sl_stop = round(last_price * (1.0 - sl_pct), 2)
    else:
        # Short: Gewinne, wenn Kurs fällt => TP unterhalb, SL oberhalb
        tp_limit = round(last_price * (1.0 - tp_pct), 2)
        sl_stop = round(last_price * (1.0 + sl_pct), 2)

    req = MarketOrderRequest(
        symbol=symbol,
        side=side,
        time_in_force=TimeInForce.DAY,
        notional=round(notional_usd, 2),
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp_limit),
        stop_loss=StopLossRequest(stop_price=sl_stop),
        extended_hours=extended,
    )
    order = trading.submit_order(order_data=req)
    print(f"[ORDER] {symbol} {side.value.upper()} notional=${notional_usd:.2f} "
          f"TP={tp_limit} SL={sl_stop} (last≈{last_price:.2f}) → id={order.id}")

def has_open_position(trading: TradingClient, symbol: str) -> bool:
    try:
        pos = trading.get_open_position(symbol)
        return pos is not None and float(pos.qty) != 0.0
    except Exception:
        return False

# ======================
# Hauptlogik
# ======================
def process_symbol(
    symbol: str,
    trading: TradingClient,
    market: StockHistoricalDataClient,
    state: Dict
) -> None:
    # Cooldown prüfen
    if not eligible_by_cooldown(state, symbol):
        return

    # Kursänderung (5m) prüfen
    df = fetch_last_minute_prices(market, symbol, minutes=6)
    chg = compute_5m_change_pct(df)
    if chg is None:
        return

    if abs(chg) < PRICE_SPIKE_PCT_5M:
        return  # keine News-Abfrage

    last_price = get_last_price_from_df(df)
    print(f"[SPIKE] {symbol}: 5m Change {chg:.2f}% • last≈{(last_price or 0):.2f} → News holen…")

    # News holen
    news_items = fetch_news_for_symbol(symbol, max_items=6)
    if not news_items:
        print(f"[NEWS] {symbol}: Keine News gefunden.")
        return

    # Dedupe/Neue Headlines
    unseen = [n for n in news_items if not already_seen(state, n["id"])]
    if not unseen:
        print(f"[NEWS] {symbol}: Nur bereits gesehene Headlines.")
        return

    # Beste (jüngste) Headline zuerst
    unseen.sort(key=lambda n: n["published"], reverse=True)

    # Position-Check
    if has_open_position(trading, symbol):
        print(f"[SKIP] {symbol}: Bereits offene Position – keine neue Order.")
        # Headlines trotzdem markieren, damit wir nicht spammen
        for n in unseen:
            mark_seen(state, n["id"])
        save_state(state)
        return

    # Sentiment prüfen & ggf. handeln (erste passende Headline nutzen)
    for n in unseen:
        title = (n.get("title") or "").strip()
        if not title:
            mark_seen(state, n["id"])
            continue

        score = sentiment_score(title)
        print(f"[SENT] {symbol}: '{title[:100]}…' → VADER {score:+.3f}")

        side: Optional[OrderSide] = None
        if score >= NEWS_POS_TH:
            side = OrderSide.BUY
        elif score <= NEWS_NEG_TH and ALLOW_SHORT:
            side = OrderSide.SELL  # Short

        # Headline immer markieren (damit sie nicht mehrfach gehandelt wird)
        mark_seen(state, n["id"])

        if side is None:
            continue  # nächste Headline probieren

        # Marktzeiten prüfen (falls gewünscht)
        if not EXTENDED_HOURS and not is_market_open(trading):
            print(f"[INFO] Markt geschlossen → kein Trade ({symbol}).")
            set_cooldown(state, symbol, COOLDOWN_MIN)  # kleiner Cooldown trotzdem
            save_state(state)
            return

        if last_price is None:
            # Notfall: falls oben nicht da, nochmal kurz ziehen
            df2 = fetch_last_minute_prices(market, symbol, minutes=2)
            last_price = get_last_price_from_df(df2)

        try:
            place_bracket_order(
                trading=trading,
                symbol=symbol,
                side=side,
                notional_usd=RISK_USD_PER_TRADE,
                last_price=last_price or 0.0,
                tp_pct=TP_PCT,
                sl_pct=SL_PCT,
                extended=EXTENDED_HOURS,
            )
            set_cooldown(state, symbol, COOLDOWN_MIN)
            save_state(state)
        except Exception as e:
            print(f"[ERR] Order fehlgeschlagen ({symbol}): {e}")
        finally:
            return  # pro Spike/Loop maximal 1 Trade je Symbol

def run_once() -> None:
    trading, market = build_clients()
    print(f"[BOT] Runde • Feed={DATA_FEED} • Watchlist={','.join(WATCHLIST)}")
    for sym in WATCHLIST:
        try:
            state = load_state()
            process_symbol(sym, trading, market, state)
        except Exception as e:
            print(f"[WARN] {sym}: {e}")

def loop_forever() -> None:
    print(f"[BOT] Starte Endlosschleife (Intervall {POLLING_SECONDS}s)")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[FATAL] {e}")
        time.sleep(POLLING_SECONDS)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="Eine Runde ausführen und beenden.")
    p.add_argument("--loop", action="store_true", help="Endlosschleife.")
    args = p.parse_args()

    if args.once:
        run_once()
        return
    if args.loop:
        loop_forever()
        return

    run_once()

if __name__ == "__main__":
    main()
