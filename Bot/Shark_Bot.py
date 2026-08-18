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
# CONFIGURATION
# ============================================================
BASE_URL = "https://api.sharkexchange.in"

SYMBOL = "BTCUSDT"          # matches the pair used in Shark's own docs examples
MARGIN_ASSET = "USDT"        # Shark Exchange margins in INR
LEVERAGE = 150                # SAFETY DEFAULT — do not raise without deliberate intent
MARGIN_MODE = "ISOLATED"    # ISOLATED caps loss to this position's margin, not the whole account

INTERVAL = "4h"
BB_PERIOD = 20
BB_STD_DEV = 2.0

SL_POINTS = 300.0
TP_POINTS = 1500.0
SL_COOLDOWN_BARS = 3

TRADE_QTY = 0.04

STATE_FILE = "shark_bot_state.json"
LOG_FILE = "shark_bot_activity.log"

# ============================================================
# SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("shark_bb_bot")

API_KEY = os.environ.get("SHARK_API_KEY")
API_SECRET = os.environ.get("SHARK_API_SECRET")
if not API_KEY or not API_SECRET:
    raise SystemExit(
        "Missing SHARK_API_KEY / SHARK_API_SECRET environment variables. "
        "Set them before running this bot."
    )


# ============================================================
# SIGNED REQUEST HELPERS (per Shark's documented auth scheme:
# HMAC-SHA256 over the query string for GET, over the JSON body for
# POST/PATCH/DELETE, sent as 'api-key' + 'signature' headers)
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
# MARKET DATA
# ============================================================
def get_closed_klines(limit=BB_PERIOD + 5):
    """
    NOTE: Shark Exchange's public kline endpoint is POST /v1/market/klines
    (confirmed from their changelog, which also mentions a `priceType`
    parameter of MARK_PRICE or LAST_PRICE). Their exact field names for
    symbol/interval/limit were not fully visible in the docs I could pull.
    VERIFY this request shape against https://docs.sharkexchange.in/
    (the "Get Klines" section under Public Endpoints) before relying on
    this function — adjust field names/response parsing to match exactly.
    """
    resp = requests.post(
        f"{BASE_URL}/v1/market/klines",
        json={
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "limit": limit,
            "priceType": "LAST_PRICE",
        },
    )
    resp.raise_for_status()
    raw = resp.json()

    # Adjust this parsing once the confirmed response schema is verified.
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
        "cooldown_until": None,
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================
# TRADING
# ============================================================
def enter_position(position_type):
    side = "SELL" if position_type == "SHORT" else "BUY"
    log.info(f"Placing {side} MARKET order for {TRADE_QTY} {SYMBOL}")

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

    # Fetch the resulting open position to get its positionId and fill price
    positions = signed_get("/v1/positions/OPEN", {"symbol": SYMBOL})
    if not positions:
        log.error("No open position found after placing entry order — check manually.")
        return None, None

    pos = positions[0]
    position_id = pos["positionId"]
    entry_price = float(pos["entryPrice"])

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

    return entry_price, position_id


def check_position_status(position_id):
    """Returns 'CLOSED', 'OPEN', or 'UNKNOWN'. Doesn't distinguish TP vs SL
    fill here — cross-reference trade history by positionId if you need
    that distinction for cooldown-only-on-SL logic."""
    try:
        positions = signed_get("/v1/positions/OPEN", {"symbol": SYMBOL})
    except Exception as e:
        log.warning(f"Could not check position status: {e}")
        return "UNKNOWN"

    still_open = any(p["positionId"] == position_id for p in positions)
    return "OPEN" if still_open else "CLOSED"


# ============================================================
# MAIN LOOP (candle-close driven, same structure as bb_live_bot.py)
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
        if state["in_position"]:
            status = check_position_status(state["position_id"])
            if status == "CLOSED":
                log.info("Position closed. Starting cooldown as a precaution "
                         "(can't yet distinguish TP vs SL fill here — see check_position_status).")
                state["cooldown_until"] = (
                    datetime.now(timezone.utc) + timedelta(hours=4 * SL_COOLDOWN_BARS)
                ).isoformat()
                state["in_position"] = False
                state["position_type"] = None
                state["position_id"] = None
                state["entry_price"] = None
                save_state(state)

        if not state["in_position"]:
            in_cooldown = (
                state["cooldown_until"]
                and datetime.now(timezone.utc) < datetime.fromisoformat(state["cooldown_until"])
            )
            if in_cooldown:
                log.info(f"In cooldown until {state['cooldown_until']}")
            else:
                df = get_closed_klines()
                df = calculate_bands(df)
                last = df.iloc[-1]

                if pd.notna(last["upper_band"]):
                    if last["high"] >= last["upper_band"]:
                        entry_price, position_id = enter_position("SHORT")
                        if position_id:
                            state.update(in_position=True, position_type="SHORT",
                                         position_id=position_id, entry_price=entry_price)
                            save_state(state)
                    elif last["low"] <= last["lower_band"]:
                        entry_price, position_id = enter_position("LONG")
                        if position_id:
                            state.update(in_position=True, position_type="LONG",
                                         position_id=position_id, entry_price=entry_price)
                            save_state(state)
                    else:
                        log.info(f"No signal. Close={last['close']:.2f}, "
                                 f"bands=[{last['lower_band']:.2f}, {last['upper_band']:.2f}]")

        sleep_s = seconds_until_next_candle_close()
        log.info(f"Sleeping {sleep_s/60:.1f} minutes until next candle close.")
        time.sleep(sleep_s)


if __name__ == "__main__":
    run()