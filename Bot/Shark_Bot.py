import os
import time
import json
import hmac
import hashlib
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

# ============================================================
# CONFIGURATION — unchanged from your verified version
# ============================================================
BASE_URL = "https://api.sharkexchange.in"

SYMBOL = "BTCUSDT"
MARGIN_ASSET = "USDT"       # confirmed against your dashboard display
LEVERAGE = 150               # left as-is, per your explicit sizing decision
MARGIN_MODE = "ISOLATED"

INTERVAL = "4h"
BB_PERIOD = 20
BB_STD_DEV = 2.0

SL_POINTS = 300.0
TP_POINTS = 1500.0
SL_COOLDOWN_BARS = 3

TRADE_QTY = 0.04

STATE_FILE = "shark_bot_state.json"
LOG_FILE = "shark_bot_activity.log"
TRADES_DIR = "trades"
EVENTS_DIR = "events"       # insufficient-funds and other non-trade events

# ============================================================
# SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("shark_bb_bot")

os.makedirs(TRADES_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)

API_KEY = os.environ.get("SHARK_API_KEY")
API_SECRET = os.environ.get("SHARK_API_SECRET")
if not API_KEY or not API_SECRET:
    raise SystemExit(
        "Missing SHARK_API_KEY / SHARK_API_SECRET environment variables. "
        "Set them before running this bot."
    )


# ============================================================
# SIGNED REQUEST HELPERS
# ============================================================
def _sign(payload_str: str) -> str:
    return hmac.new(API_SECRET.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256).hexdigest()


def signed_get(endpoint: str, params: dict = None) -> dict:
    params = dict(params or {})
    params["timestamp"] = str(int(time.time() * 1000))
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    signature = _sign(query_string)
    headers = {"api-key": API_KEY, "signature": signature}
    resp = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


def signed_post(endpoint: str, params: dict = None) -> dict:
    params = dict(params or {})
    params["timestamp"] = str(int(time.time() * 1000))
    data_to_sign = json.dumps(params, separators=(",", ":"))
    signature = _sign(data_to_sign)
    headers = {"api-key": API_KEY, "signature": signature, "Content-Type": "application/json"}
    resp = requests.post(f"{BASE_URL}{endpoint}", json=params, headers=headers)
    resp.raise_for_status()
    return resp.json()


# ============================================================
# MARKET DATA — still flagged, verify field names against docs
# ============================================================
def get_closed_klines(limit=BB_PERIOD + 5):
    """UNVERIFIED field names — confirm against docs.sharkexchange.in
    ('Get Klines' under Public Endpoints) before trusting this in live use."""
    resp = requests.post(
        f"{BASE_URL}/v1/market/klines",
        json={"symbol": SYMBOL, "interval": INTERVAL, "limit": limit, "priceType": "LAST_PRICE"},
    )
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw)
    df.columns = [c.lower() for c in df.columns]
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

    now = datetime.now(timezone.utc)
    df = df[df["close_time"] <= now].reset_index(drop=True)
    return df


def calculate_bands(df):
    df["sma"] = df["close"].rolling(window=BB_PERIOD).mean()
    df["std"] = df["close"].rolling(window=BB_PERIOD).std()
    df["upper_band"] = df["sma"] + df["std"] * BB_STD_DEV
    df["lower_band"] = df["sma"] - df["std"] * BB_STD_DEV
    return df


# ============================================================
# ACCOUNT SETUP
# ============================================================
def set_leverage_and_margin_mode():
    result = signed_post(
        "/v1/exchange/update/preference",
        {"leverage": LEVERAGE, "marginMode": MARGIN_MODE, "contractName": SYMBOL},
    )
    log.info(f"Leverage/margin mode set: {result}")


def get_available_balance():
    """
    UNVERIFIED ENDPOINT: I have not confirmed Shark Exchange's exact wallet
    /ballance endpoint path or response field name. This assumes something
    like GET /v1/wallet/balance -> {"availableBalance": "123.45", ...}.
    VERIFY this against their docs or by testing manually before relying on
    it — if the field name is wrong, this will raise, which is at least a
    safe failure mode (bot logs the error and skips the trade rather than
    placing an order blind).
    """
    try:
        resp = signed_get("/v1/wallet/balance", {"asset": MARGIN_ASSET})
        return float(resp.get("availableBalance", resp.get("balance", 0.0)))
    except Exception as e:
        log.error(f"Could not fetch available balance (endpoint unverified — check docs): {e}")
        return None


def compute_margin_used(entry_price, qty, leverage):
    return round((qty * entry_price) / leverage, 4)


# ============================================================
# STATE
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "in_position": False,
        "position_type": None,
        "position_id": None,
        "entry_price": None,
        "entry_time": None,
        "sl_price": None,
        "tp_price": None,
        "cooldown_until": None,
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================
# JSON LOGGING — trades and non-trade events (e.g. insufficient funds)
# ============================================================
def write_trade_json(state, exit_price, exit_reason):
    entry_price = state["entry_price"]
    position_type = state["position_type"]
    qty = TRADE_QTY
    sl_price = state["sl_price"]
    tp_price = state["tp_price"]

    points = (exit_price - entry_price) if position_type == "LONG" else (entry_price - exit_price)
    pnl_usd = round(points * qty, 4)
    margin_used = compute_margin_used(entry_price, qty, LEVERAGE)
    pnl_pct_on_margin = round((pnl_usd / margin_used) * 100, 2) if margin_used else 0.0

    # Risk:Reward
    risk_points = abs(entry_price - sl_price) if sl_price else SL_POINTS
    reward_points = abs(tp_price - entry_price) if tp_price else TP_POINTS
    planned_rr = round(reward_points / risk_points, 2) if risk_points else 0.0
    # Realized R:R — how many multiples of your risk you actually captured
    # (negative if it was a loss)
    realized_rr = round(points / risk_points, 2) if risk_points else 0.0

    exit_time = datetime.now(timezone.utc)
    trade_record = {
        "position_id": state["position_id"],
        "symbol": SYMBOL,
        "type": position_type,
        "quantity": qty,
        "leverage": LEVERAGE,
        "margin_asset": MARGIN_ASSET,
        "margin_used": margin_used,
        "entry_time": state["entry_time"],
        "exit_time": exit_time.isoformat(),
        "entry_price": entry_price,
        "exit_price": round(exit_price, 2),
        "points_captured": round(points, 2),
        "pnl_usd": pnl_usd,
        "pnl_pct_on_margin": pnl_pct_on_margin,
        "planned_risk_reward": planned_rr,
        "realized_risk_reward": realized_rr,
        "exit_reason": exit_reason,
    }

    filename = f"{TRADES_DIR}/trade_{state['position_id']}_{exit_time.strftime('%Y%m%dT%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(trade_record, f, indent=2)

    log.info(f"Trade logged to {filename}: {exit_reason}, pnl_usd={pnl_usd}, R:R={realized_rr}")
    return trade_record


def write_insufficient_funds_json(signal_type, required_margin, available_balance, last_price):
    """Logged whenever a valid entry signal fires but there isn't enough
    balance to open it. The bot does NOT stop — it logs this and keeps
    checking on future candles in case funds are added later."""
    event_time = datetime.now(timezone.utc)
    record = {
        "event": "INSUFFICIENT_FUNDS",
        "timestamp": event_time.isoformat(),
        "signal_type": signal_type,
        "symbol": SYMBOL,
        "attempted_quantity": TRADE_QTY,
        "leverage": LEVERAGE,
        "margin_asset": MARGIN_ASSET,
        "required_margin": required_margin,
        "available_balance": available_balance,
        "shortfall": round(required_margin - available_balance, 4) if available_balance is not None else None,
        "last_known_price": round(last_price, 2),
    }

    filename = f"{EVENTS_DIR}/insufficient_funds_{event_time.strftime('%Y%m%dT%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(record, f, indent=2)

    log.warning(f"Insufficient funds — signal skipped. Logged to {filename}")
    return record


# ============================================================
# TRADING
# ============================================================
def enter_position(position_type, last_price):
    # --- Funds check before placing any order ---
    required_margin = compute_margin_used(last_price, TRADE_QTY, LEVERAGE)
    available_balance = get_available_balance()

    if available_balance is None:
        log.error("Skipping entry — could not verify balance (see get_available_balance).")
        return None

    if available_balance < required_margin:
        write_insufficient_funds_json(position_type, required_margin, available_balance, last_price)
        return None

    side = "SELL" if position_type == "SHORT" else "BUY"
    log.info(f"Placing {side} MARKET order for {TRADE_QTY} {SYMBOL} "
             f"(required margin ~{required_margin}, available {available_balance})")

    order = signed_post(
        "/v1/order/place-order",
        {
            "placeType": "ORDER_FORM",
            "quantity": TRADE_QTY,
            "side": side,
            "symbol": SYMBOL,
            "type": "MARKET",
            "reduceOnly": False,
            "marginAsset": MARGIN_ASSET,
        },
    )
    log.info(f"Entry order response: {order}")

    positions = signed_get("/v1/positions/OPEN", {"symbol": SYMBOL})
    if not positions:
        log.error("No open position found after placing entry order — check manually.")
        return None

    pos = positions[0]
    position_id = pos["positionId"]
    entry_price = float(pos["entryPrice"])
    entry_time = datetime.now(timezone.utc).isoformat()

    if position_type == "LONG":
        sl_price = round(entry_price - SL_POINTS, 2)
        tp_price = round(entry_price + TP_POINTS, 2)
    else:
        sl_price = round(entry_price + SL_POINTS, 2)
        tp_price = round(entry_price - TP_POINTS, 2)

    tp_sl_result = signed_post(
        "/v2/order/split-tp-sl",
        {
            "positionId": position_id,
            "splitTakeProfitOrders": [{"quantity": TRADE_QTY, "price": tp_price}],
            "splitStopLossOrders": [{"quantity": TRADE_QTY, "price": sl_price}],
        },
    )
    log.info(f"SL/TP attached: TP={tp_price} SL={sl_price} — {tp_sl_result}")

    return {
        "position_id": position_id,
        "entry_price": entry_price,
        "entry_time": entry_time,
        "sl_price": sl_price,
        "tp_price": tp_price,
    }


def close_position_market(position_type):
    side = "BUY" if position_type == "SHORT" else "SELL"
    log.info(f"Middle Band touched — closing {position_type} with {side} MARKET reduceOnly order")
    result = signed_post(
        "/v1/order/place-order",
        {
            "placeType": "ORDER_FORM",
            "quantity": TRADE_QTY,
            "side": side,
            "symbol": SYMBOL,
            "type": "MARKET",
            "reduceOnly": True,
            "marginAsset": MARGIN_ASSET,
        },
    )
    log.info(f"Middle Band close order response: {result}")
    return result


def check_position_status(position_id):
    try:
        positions = signed_get("/v1/positions/OPEN", {"symbol": SYMBOL})
    except Exception as e:
        log.warning(f"Could not check position status: {e}")
        return "UNKNOWN"
    still_open = any(p["positionId"] == position_id for p in positions)
    return "OPEN" if still_open else "CLOSED"


def determine_exit_reason(state, last_price):
    """Heuristic — see earlier note: not a confirmed read of Shark's own
    trade record, since that endpoint's schema isn't verified."""
    sl_price = state["sl_price"]
    tp_price = state["tp_price"]
    position_type = state["position_type"]

    if position_type == "LONG":
        hit_sl = last_price <= sl_price
        hit_tp = last_price >= tp_price
    else:
        hit_sl = last_price >= sl_price
        hit_tp = last_price <= tp_price

    if hit_sl:
        return "Stop Loss", sl_price
    if hit_tp:
        return "Take Profit", tp_price
    return "Middle Band", last_price


# ============================================================
# MAIN LOOP
# ============================================================
def seconds_until_next_candle_close():
    now = datetime.now(timezone.utc)
    hours_since = now.hour % 4
    next_close = (now + timedelta(hours=4 - hours_since)).replace(minute=0, second=0, microsecond=0)
    if next_close <= now:
        next_close += timedelta(hours=4)
    return (next_close - now).total_seconds() + 15


def run():
    set_leverage_and_margin_mode()
    state = load_state()
    log.info("Shark Exchange bot started. State: %s", state)

    while True:
        df = get_closed_klines()
        df = calculate_bands(df)
        last = df.iloc[-1]

        if state["in_position"]:
            status = check_position_status(state["position_id"])

            if status == "CLOSED":
                reason, exit_price = determine_exit_reason(state, last["close"])
                write_trade_json(state, exit_price, reason)

                if reason == "Stop Loss":
                    state["cooldown_until"] = (
                        datetime.now(timezone.utc) + timedelta(hours=4 * SL_COOLDOWN_BARS)
                    ).isoformat()
                    log.info(f"Stop-loss hit — cooldown until {state['cooldown_until']}")
                else:
                    state["cooldown_until"] = None
                    log.info(f"{reason} exit — no cooldown, can re-enter next signal.")

                state.update(in_position=False, position_type=None, position_id=None,
                             entry_price=None, entry_time=None, sl_price=None, tp_price=None)
                save_state(state)

            else:
                pt = state["position_type"]
                mid_hit = (
                    (pt == "SHORT" and last["low"] <= last["sma"]) or
                    (pt == "LONG" and last["high"] >= last["sma"])
                )
                if mid_hit:
                    close_position_market(pt)
                    write_trade_json(state, last["sma"], "Middle Band")
                    state.update(in_position=False, position_type=None, position_id=None,
                                 entry_price=None, entry_time=None, sl_price=None, tp_price=None,
                                 cooldown_until=None)
                    save_state(state)

        if not state["in_position"]:
            in_cooldown = (
                state["cooldown_until"]
                and datetime.now(timezone.utc) < datetime.fromisoformat(state["cooldown_until"])
            )
            if in_cooldown:
                log.info(f"In cooldown until {state['cooldown_until']}")
            elif pd.notna(last["upper_band"]):
                if last["high"] >= last["upper_band"]:
                    entry = enter_position("SHORT", last["close"])
                    if entry:
                        state.update(in_position=True, position_type="SHORT", **entry)
                        save_state(state)
                elif last["low"] <= last["lower_band"]:
                    entry = enter_position("LONG", last["close"])
                    if entry:
                        state.update(in_position=True, position_type="LONG", **entry)
                        save_state(state)
                else:
                    log.info(f"No signal. Close={last['close']:.2f}, "
                             f"bands=[{last['lower_band']:.2f}, {last['upper_band']:.2f}]")

        sleep_s = seconds_until_next_candle_close()
        log.info(f"Sleeping {sleep_s/60:.1f} minutes until next candle close.")
        time.sleep(sleep_s)


if __name__ == "__main__":
    run()