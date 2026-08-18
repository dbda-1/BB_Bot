import math
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt


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
    """
    trades = []
    in_position = False
    position_type = None  # 'LONG' or 'SHORT'
    entry_price = 0.0
    entry_index = 0
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
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Stop Loss",
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
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Take Profit",
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
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Middle Band",
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
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Stop Loss",
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
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Take Profit",
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
                            "pnl_pct": pnl_pct,
                            "bars": bars_held,
                            "reason": "Middle Band",
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

            # LONG Entry: Price touches or exceeds Lower Band
            elif row["low"] <= row["lower_band"]:
                in_position = True
                position_type = "LONG"
                entry_price = row["lower_band"]
                entry_index = i

    return pd.DataFrame(trades)


def compute_streaks(pnl_series):
    """Max consecutive winning trades in a row, and max consecutive losing
    trades in a row, walking through trades in chronological order."""
    current_win_streak = 0
    current_loss_streak = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0

    for pnl in pnl_series:
        if pnl > 0:
            current_win_streak += 1
            current_loss_streak = 0
        else:
            current_loss_streak += 1
            current_win_streak = 0
        max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
        max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)

    return max_consecutive_wins, max_consecutive_losses


def plot_monthly_capital_growth(
    trades_df,
    initial_capital=10000.0,
    fee_rate_per_side=0.0004,   # 0.04% per side -> 0.08% round trip
    save_path="monthly_capital_growth.png",
):
    """Plot capital growth by month, compounding each trade's pnl_pct in
    chronological order (one position at a time, so this is a valid
    sequential equity curve) — net of round-trip trading fees.

    Every trade pays the round-trip fee regardless of whether it was a
    win or a loss, so it's subtracted from pnl_pct before compounding.
    """
    round_trip_fee = fee_rate_per_side * 2

    tdf = trades_df.sort_values("exit_time").reset_index(drop=True).copy()
    tdf["net_pnl_pct"] = tdf["pnl_pct"] - round_trip_fee

    tdf["gross_capital"] = initial_capital * (1 + tdf["pnl_pct"]).cumprod()
    tdf["net_capital"] = initial_capital * (1 + tdf["net_pnl_pct"]).cumprod()

    monthly_gross = tdf.set_index("exit_time")["gross_capital"].resample("ME").last().ffill()
    monthly_net = tdf.set_index("exit_time")["net_capital"].resample("ME").last().ffill()

    # Prepend starting capital so the chart shows growth from day one
    first_month_start = monthly_net.index[0] - pd.offsets.MonthEnd(1)
    monthly_gross = pd.concat([pd.Series([initial_capital], index=[first_month_start]), monthly_gross])
    monthly_net = pd.concat([pd.Series([initial_capital], index=[first_month_start]), monthly_net])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(monthly_gross.index, monthly_gross.values, marker="o", linewidth=1.5,
            color="#94a3b8", linestyle="--", label="Gross capital (before fees)")
    ax.plot(monthly_net.index, monthly_net.values, marker="o", linewidth=2.5,
            color="#16a34a", label="Net capital (after 0.08% round-trip fees)")
    ax.fill_between(monthly_net.index, initial_capital, monthly_net.values, alpha=0.08, color="#16a34a")
    ax.axhline(initial_capital, color="gray", linestyle=":", linewidth=1, label="Starting capital")

    ax.set_title("Capital Growth by Month — Gross vs. Net of Fees", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Capital ($)")
    ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"📈 Saved fee-adjusted capital growth chart to {save_path}")
    print(f"   Gross ending capital : ${monthly_gross.iloc[-1]:,.2f}")
    print(f"   Net ending capital   : ${monthly_net.iloc[-1]:,.2f}")
    print(f"   Fee drag (gross-net) : ${monthly_gross.iloc[-1] - monthly_net.iloc[-1]:,.2f}")
    print(f"   Round-trip fee rate  : {round_trip_fee*100:.2f}% per trade  ({len(tdf)} trades)\n")

    return monthly_gross, monthly_net


def run_and_print_results():
    # --- USER CONFIGURATION ---
    symbol = "BTCUSDT"
    timeframe = "4h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    bb_period = 20
    bb_std_dev = 2.0
    sl_points = 300.0  # $300 SL
    tp_points = 1500.0  # TP points
    sl_cooldown_bars = 3  # candles to wait after an SL hit before next trade
    initial_capital = 10000.0  # starting capital for the equity curve / chart
    fee_rate_per_side = 0.0004  # 0.04% per side -> 0.08% round trip

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
    print(f"   Fee per side : {fee_rate_per_side*100:.2f}%  ({fee_rate_per_side*2*100:.2f}% round trip)")
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

    # Consecutive Wins/Losses (in a row, chronological order)
    max_consecutive_wins, max_consecutive_losses = compute_streaks(trades_df["pnl_pct"])

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

    # Output Printing
    print("=" * 68)
    print("📊 BOLLINGER BANDS MEAN REVERSION - BACKTEST RESULTS")
    print("=" * 68)
    print("\n📈 PERFORMANCE METRICS:")
    print(f"   Total Trades    : {total_trades}")
    print(f"   Winning Trades  : {winning_trades}")
    print(f"   Losing Trades   : {losing_trades}")
    print(f"   Win Rate        : {win_rate:.1f}%")
    print(f"   Total P&L (gross): {total_pnl:+.2f}%")
    print(f"   Avg Win         : {avg_win:+.2f}%")
    print(f"   Avg Loss        : {avg_loss:+.2f}%")
    print(f"   Risk/Reward     : {risk_reward:.2f}")
    print(f"   Profit Factor   : {profit_factor:.2f}")
    print(f"   Max Drawdown    : {max_drawdown:.2f}%")

    print("\n🎲 RISK METRICS:")
    print(f"   Max Consecutive Wins  : {max_consecutive_wins}")
    print(f"   Max Consecutive Losses: {max_consecutive_losses}")
    print(f"   Avg Bars Held         : {avg_bars_held:.1f}")
    print(f"   Sharpe Ratio          : {sharpe_ratio:.1f}")

    print("\n🏆 BEST & WORST:")
    print(f"   Best Trade   : {best_trade:+.2f}%")
    print(f"   Worst Trade  : {worst_trade:+.2f}%")

    print("\n🚪 EXIT REASONS:")
    print(f"   Stop Loss: {sl_count}")
    print(f"   Middle Band: {mid_count}")
    print(f"   Take Profit: {tp_count}\n")

    # Monthly capital growth chart, net of fees
    plot_monthly_capital_growth(
        trades_df,
        initial_capital=initial_capital,
        fee_rate_per_side=fee_rate_per_side,
    )


if __name__ == "__main__":
    run_and_print_results()