import math
from datetime import datetime
import numpy as np
import pandas as pd
import requests


def fetch_binance_klines(symbol="BTCUSDT", interval="4h", start_str="2025-04-24", end_str="2026-04-24"):
    """Fetch historical kline data from Binance API without needing an API key."""
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)

    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    current_start = start_ts

    print(f"📥 Fetching {symbol} {interval} data for date range...")
    print(f"   From: {start_str}")
    print(f"   To:   {end_str}")

    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000,
        }
        res = requests.get(url, params=params)
        data = res.json()

        if not data or not isinstance(data, list):
            break

        all_candles.extend(data)
        current_start = data[-1][0] + 1

        if len(data) < 1000:
            break

    print(f"   Downloaded {len(all_candles)} candles...")
    print(f"   ✅ Downloaded {len(all_candles)} candles\n")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qav",
        "num_trades",
        "tb_base",
        "tb_quote",
        "ignore",
    ]
    df = pd.DataFrame(all_candles, columns=cols)

    # Clean data types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def calculate_bollinger_bands(df, period=20, std_dev=2.0):
    """Calculate Bollinger Bands (Middle, Upper, Lower)."""
    df["sma"] = df["close"].rolling(window=period).mean()
    df["std"] = df["close"].rolling(window=period).std()
    df["upper_band"] = df["sma"] + (df["std"] * std_dev)
    df["lower_band"] = df["sma"] - (df["std"] * std_dev)
    return df


def backtest_strategy(df, sl_points=300.0, tp_points=1500.0, sl_cooldown_bars=3):
    """Backtest Bollinger Bands Mean Reversion Strategy.

    - Sell (Short) when price touches Upper Band
    - Buy (Long) when price touches Lower Band
    - SL: $300 points
    - TP: $1,500 points OR Price touches Middle Band

    Cooldown rule: if a trade is closed by Stop Loss, no new trade can be
    opened for `sl_cooldown_bars` candles after that exit. Trades that close
    via Take Profit or Middle Band are NOT subject to any cooldown — the
    very next qualifying bar can open a new trade as before.

    Each closed trade also records `entry_time` and `exit_time` so trades
    can be grouped and analyzed by month afterward, plus `points` — the
    raw price distance captured on the trade (same sign convention as
    pnl_pct: positive = profit, negative = loss).
    """
    trades = []
    in_position = False
    position_type = None  # 'LONG' or 'SHORT'
    entry_price = 0.0
    entry_index = 0
    entry_time = None
    sl_block_until_index = -1  # no entries allowed until index > this

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Skip warm-up period for Bollinger Bands
        if pd.isna(prev_row["upper_band"]):
            continue

        if in_position:
            bars_held = i - entry_index

            if position_type == "SHORT":
                sl_price = entry_price + sl_points
                tp_price = entry_price - tp_points

                # 1. Check Stop Loss ($300 above entry)
                if row["high"] >= sl_price:
                    exit_price = sl_price
                    pnl_pct = (entry_price - exit_price) / entry_price
                    trades.append(
                        {
                            "type": "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "points": entry_price - exit_price,
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Stop Loss",
                            "entry_time": entry_time,
                            "exit_time": row["open_time"],
                        }
                    )
                    in_position = False
                    sl_block_until_index = i + sl_cooldown_bars

                # 2. Check Target TP ($1,500 below entry)
                elif row["low"] <= tp_price:
                    exit_price = tp_price
                    pnl_pct = (entry_price - exit_price) / entry_price
                    trades.append(
                        {
                            "type": "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "points": entry_price - exit_price,
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Take Profit",
                            "entry_time": entry_time,
                            "exit_time": row["open_time"],
                        }
                    )
                    in_position = False

                # 3. Check Middle Band Exit
                elif row["low"] <= row["sma"]:
                    exit_price = row["sma"]
                    pnl_pct = (entry_price - exit_price) / entry_price
                    trades.append(
                        {
                            "type": "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "points": entry_price - exit_price,
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Middle Band",
                            "entry_time": entry_time,
                            "exit_time": row["open_time"],
                        }
                    )
                    in_position = False

            elif position_type == "LONG":
                sl_price = entry_price - sl_points
                tp_price = entry_price + tp_points

                # 1. Check Stop Loss ($300 below entry)
                if row["low"] <= sl_price:
                    exit_price = sl_price
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trades.append(
                        {
                            "type": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "points": exit_price - entry_price,
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Stop Loss",
                            "entry_time": entry_time,
                            "exit_time": row["open_time"],
                        }
                    )
                    in_position = False
                    sl_block_until_index = i + sl_cooldown_bars

                # 2. Check Target TP ($1,500 above entry)
                elif row["high"] >= tp_price:
                    exit_price = tp_price
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trades.append(
                        {
                            "type": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "points": exit_price - entry_price,
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Take Profit",
                            "entry_time": entry_time,
                            "exit_time": row["open_time"],
                        }
                    )
                    in_position = False

                # 3. Check Middle Band Exit
                elif row["high"] >= row["sma"]:
                    exit_price = row["sma"]
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trades.append(
                        {
                            "type": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "points": exit_price - entry_price,
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Middle Band",
                            "entry_time": entry_time,
                            "exit_time": row["open_time"],
                        }
                    )
                    in_position = False

        else:
            # Respect SL cooldown: no new entries until this many bars pass
            if i <= sl_block_until_index:
                continue

            # Check for Entry Triggers
            # SHORT Entry: Price touches or exceeds Upper Band
            if row["high"] >= row["upper_band"]:
                in_position = True
                position_type = "SHORT"
                entry_price = row["upper_band"]
                entry_index = i
                entry_time = row["open_time"]

            # LONG Entry: Price touches or exceeds Lower Band
            elif row["low"] <= row["lower_band"]:
                in_position = True
                position_type = "LONG"
                entry_price = row["lower_band"]
                entry_index = i
                entry_time = row["open_time"]

    return pd.DataFrame(trades)


def _max_streak(bool_series):
    """Longest run of consecutive True values in a boolean Series."""
    if bool_series.empty:
        return 0
    groups = (bool_series != bool_series.shift()).cumsum()
    run_lengths = bool_series.groupby(groups).cumsum()
    return int(run_lengths[bool_series].max()) if bool_series.any() else 0


def print_detailed_monthly_analysis(trades_df):
    """Print a very detailed, per-month breakdown of trading performance:
    trade counts, win rate, P&L stats, avg win/loss, best/worst trade,
    profit factor, risk/reward, in-month streaks, avg bars held, exit
    reason mix, and a running cumulative P&L total across the whole
    selected period."""
    tdf = trades_df.copy()
    tdf["month"] = tdf["exit_time"].dt.to_period("M")

    months = sorted(tdf["month"].unique())
    cumulative_pnl = 0.0

    print("=" * 78)
    print("📅 DETAILED MONTHLY ANALYSIS")
    print("=" * 78)

    summary_rows = []

    for month in months:
        m = tdf[tdf["month"] == month].reset_index(drop=True)

        total = len(m)
        wins = m[m["pnl_pct"] > 0]
        losses = m[m["pnl_pct"] <= 0]
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / total * 100) if total > 0 else 0

        total_pnl_pct = m["pnl_pct"].sum() * 100
        avg_win_pct = wins["pnl_pct"].mean() * 100 if n_wins > 0 else 0
        avg_loss_pct = losses["pnl_pct"].mean() * 100 if n_losses > 0 else 0

        best_trade_pct = m["pnl_pct"].max() * 100
        worst_trade_pct = m["pnl_pct"].min() * 100

        gross_win = wins["pnl_pct"].sum()
        gross_loss = abs(losses["pnl_pct"].sum())
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0
        risk_reward = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0

        avg_bars_held = m["bars"].mean()

        is_win_month = m["pnl_pct"] > 0
        max_win_streak_month = _max_streak(is_win_month)
        max_loss_streak_month = _max_streak(~is_win_month)

        sl_ct = (m["reason"] == "Stop Loss").sum()
        mid_ct = (m["reason"] == "Middle Band").sum()
        tp_ct = (m["reason"] == "Take Profit").sum()

        long_ct = (m["type"] == "LONG").sum()
        short_ct = (m["type"] == "SHORT").sum()

        cumulative_pnl += total_pnl_pct

        print(f"\n{'-' * 78}")
        print(f"📆 {month}")
        print(f"{'-' * 78}")
        print(f"   Trades           : {total}   (LONG: {long_ct}, SHORT: {short_ct})")
        print(f"   Wins / Losses    : {n_wins} / {n_losses}")
        print(f"   Win Rate         : {win_rate:.1f}%")
        print(f"   Total P&L        : {total_pnl_pct:+.2f}%")
        print(f"   Cumulative P&L   : {cumulative_pnl:+.2f}%  (running total to end of this month)")
        print(f"   Avg Win / Loss   : {avg_win_pct:+.2f}% / {avg_loss_pct:+.2f}%")
        print(f"   Best / Worst     : {best_trade_pct:+.2f}% / {worst_trade_pct:+.2f}%")
        if profit_factor == float("inf"):
            print(f"   Profit Factor    : inf (no losses this month)")
        else:
            print(f"   Profit Factor    : {profit_factor:.2f}")
        print(f"   Risk/Reward      : {risk_reward:.2f}")
        print(f"   Avg Bars Held    : {avg_bars_held:.1f}")
        print(f"   Max Win Streak   : {max_win_streak_month} (within month)")
        print(f"   Max Loss Streak  : {max_loss_streak_month} (within month)")
        print(f"   Exit Reasons     : SL={sl_ct}  Middle Band={mid_ct}  TP={tp_ct}")

        summary_rows.append(
            {
                "month": str(month),
                "trades": total,
                "win_rate": win_rate,
                "pnl_pct": total_pnl_pct,
                "cum_pnl_pct": cumulative_pnl,
                "profit_factor": profit_factor,
                "sl": sl_ct, "mid": mid_ct, "tp": tp_ct,
            }
        )

    # --- Compact summary table across all months ---
    print(f"\n{'=' * 78}")
    print("📋 MONTHLY SUMMARY TABLE")
    print("=" * 78)
    header = f"{'Month':<10}{'Trades':>8}{'WinRt%':>9}{'P&L%':>10}{'CumP&L%':>11}{'PF':>7}{'SL':>5}{'Mid':>5}{'TP':>5}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        print(
            f"{r['month']:<10}{r['trades']:>8}{r['win_rate']:>8.1f}%{r['pnl_pct']:>+9.2f}%"
            f"{r['cum_pnl_pct']:>+10.2f}%{pf_str:>7}{r['sl']:>5}{r['mid']:>5}{r['tp']:>5}"
        )
    print("-" * len(header))

    total_trades_all = tdf.shape[0]
    total_pnl_all = tdf["pnl_pct"].sum() * 100
    print(f"\nTOTAL across {len(months)} months: {total_trades_all} trades, {total_pnl_all:+.2f}% combined P&L\n")


def run_and_print_results():
    # --- USER CONFIGURATION ---
    symbol = "BTCUSDT"
    timeframe = "4h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    bb_period = 20
    bb_std_dev = 2.0
    sl_points = 300.0  # $300 SL
    tp_points = 1500.0  # $1,500 TP
    sl_cooldown_bars = 3  # candles to wait after an SL hit before next trade

    # Print Configuration Section
    print("=" * 68)
    print(" BOLLINGER BANDS MEAN REVERSION BACKTEST")
    print("=" * 68)
    print("\n📋 CONFIGURATION:")
    print(f"   Symbol       : {symbol}")
    print(f"   Timeframe    : {timeframe}")
    print(f"   Start Date   : {start_date}")
    print(f"   End Date     : {end_date}")
    print(f"   BB Period    : {bb_period}")
    print(f"   BB Std Dev   : {bb_std_dev}")
    print(f"   Stop Loss    : ${sl_points:,.1f}")
    print(f"   Take Profit  : ${tp_points:,.1f} or Middle Band")
    print(f"   SL Cooldown  : {sl_cooldown_bars} candles after a Stop Loss hit")
    print()

    # Data Fetching
    df = fetch_binance_klines(
        symbol=symbol,
        interval=timeframe,
        start_str=start_date,
        end_str=end_date,
    )

    min_date_str = df["open_time"].min().strftime("%Y-%m-%d %H:%M:%S")
    max_date_str = df["open_time"].max().strftime("%Y-%m-%d %H:%M:%S")
    min_price = df["low"].min()
    max_price = df["high"].max()

    print(f"📊 Data shape: {len(df)} rows, {df.shape[1]} columns")
    print(f"   Date range: {min_date_str} to {max_date_str}")
    print(f"   Price range: ${min_price:,.2f} to ${max_price:,.2f}\n")

    print("🔄 Running backtest...\n")

    # Strategy Calculations
    df = calculate_bollinger_bands(
        df, period=bb_period, std_dev=bb_std_dev
    )
    trades_df = backtest_strategy(
        df, sl_points=sl_points, tp_points=tp_points, sl_cooldown_bars=sl_cooldown_bars
    )

    if trades_df.empty:
        print("No trades generated during this period.")
        return

    # Metrics Calculation
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["pnl_pct"] > 0])
    losing_trades = len(trades_df[trades_df["pnl_pct"] <= 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    total_pnl = trades_df["pnl_pct"].sum() * 100
    avg_win = (
        trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"].mean() * 100
        if winning_trades > 0
        else 0
    )
    avg_loss = (
        trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"].mean() * 100
        if losing_trades > 0
        else 0
    )

    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    gross_win = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"].sum()
    gross_loss = abs(trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0

    # Drawdown calculation
    cum_returns = (1 + trades_df["pnl_pct"]).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

    # Consecutive Wins/Losses
    is_win = (trades_df["pnl_pct"] > 0).astype(int)
    streaks = is_win.groupby((is_win != is_win.shift()).cumsum()).cumsum()
    max_consecutive_wins = streaks[is_win == 1].max() if 1 in is_win.values else 0
    max_consecutive_losses = (
        (is_win == 0)
        .groupby(((is_win == 0) != (is_win == 0).shift()).cumsum())
        .cumsum()
        .max()
    )

    avg_bars_held = trades_df["bars"].mean()
    sharpe_ratio = (
        (trades_df["pnl_pct"].mean() / trades_df["pnl_pct"].std())
        * math.sqrt(252)
        if trades_df["pnl_pct"].std() > 0
        else 0
    )

    best_trade = trades_df["pnl_pct"].max() * 100
    worst_trade = trades_df["pnl_pct"].min() * 100

    sl_count = len(trades_df[trades_df["reason"] == "Stop Loss"])
    mid_count = len(trades_df[trades_df["reason"] == "Middle Band"])
    tp_count = len(trades_df[trades_df["reason"] == "Take Profit"])

    # --- Middle Band points captured ---
    mid_band_trades = trades_df[trades_df["reason"] == "Middle Band"]
    if not mid_band_trades.empty:
        mid_total_points = mid_band_trades["points"].sum()
        mid_avg_points = mid_band_trades["points"].mean()
        mid_best_points = mid_band_trades["points"].max()
        mid_worst_points = mid_band_trades["points"].min()
    else:
        mid_total_points = mid_avg_points = mid_best_points = mid_worst_points = 0.0

    # Output Printing
    print("=" * 68)
    print("📊 BOLLINGER BANDS MEAN REVERSION - BACKTEST RESULTS")
    print("=" * 68)
    print("\n📈 PERFORMANCE METRICS:")
    print(f"   Total Trades    : {total_trades}")
    print(f"   Winning Trades  : {winning_trades}")
    print(f"   Losing Trades   : {losing_trades}")
    print(f"   Win Rate        : {win_rate:.1f}%")
    print(f"   Total P&L       : {total_pnl:+.2f}%")
    print(f"   Avg Win         : {avg_win:+.2f}%")
    print(f"   Avg Loss        : {avg_loss:+.2f}%")
    print(f"   Risk/Reward     : {risk_reward:.2f}")
    print(f"   Profit Factor   : {profit_factor:.2f}")
    print(f"   Max Drawdown    : {max_drawdown:.2f}%")

    print("\n🎲 RISK METRICS:")
    print(f"   Max Consecutive Wins : {max_consecutive_wins}")
    print(f"   Max Consecutive Losses: {max_consecutive_losses}")
    print(f"   Avg Bars Held        : {avg_bars_held:.1f}")
    print(f"   Sharpe Ratio         : {sharpe_ratio:.1f}")

    print("\n🏆 BEST & WORST:")
    print(f"   Best Trade   : {best_trade:+.2f}%")
    print(f"   Worst Trade  : {worst_trade:+.2f}%")

    print("\n🚪 EXIT REASONS:")
    print(f"   Stop Loss: {sl_count}")
    print(f"   Middle Band: {mid_count}")
    print(f"   Take Profit: {tp_count}")
    print(f"\n   Middle Band Points Captured:")
    print(f"      Total  : {mid_total_points:+,.2f}")
    print(f"      Avg    : {mid_avg_points:+,.2f}")
    print(f"      Best   : {mid_best_points:+,.2f}")
    print(f"      Worst  : {mid_worst_points:+,.2f}\n")

    print_detailed_monthly_analysis(trades_df)


if __name__ == "__main__":
    run_and_print_results()