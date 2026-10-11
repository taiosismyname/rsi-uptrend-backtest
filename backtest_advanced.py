"""
Advanced RSI Uptrend Backtest Framework
========================================
Includes:
1. Buy-and-Hold Benchmark Comparison
2. CSV Trade Log Export
3. Parameter Sweep Optimization
4. Multi-Ticker Basket Testing

All edge cases handled:
- yfinance MultiIndex flattening
- PercentSizer for capital deployment
- Safe analyzer extraction (handles zero trades)
- Modular strategy definitions
"""

import os
from datetime import datetime, timedelta

import backtrader as bt
import pandas as pd
import yfinance as yf


# ==========================================
# STRATEGY DEFINITIONS
# ==========================================
class RSIUptrendStrategy(bt.Strategy):
    """
    Buy when RSI drops below threshold while price is above moving average.
    Sell when RSI reaches overbought or price closes below exit MA.
    """

    params = (
        ("rsi_period", 14),
        ("rsi_threshold", 30),
        ("uptrend_ma_period", 50),
        ("rsi_overbought", 70),
        ("exit_ma_period", 20),
        ("log_trades", True),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.uptrend_ma_period
        )
        self.exit_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.exit_ma_period
        )

        self.order = None
        self.trade_logs = []
        self.buy_price = None

    def notify_order(self, order):
        """Capture order executions and log trades."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            dt_str = self.datas[0].datetime.date(0).isoformat()
            order_type = "BUY" if order.isbuy() else "SELL"
            exec_price = order.executed.price
            pnl = order.executed.pnl if order.issell() else 0.0

            if order.isbuy():
                self.buy_price = exec_price

            if self.params.log_trades:
                self.trade_logs.append(
                    {
                        "Date": dt_str,
                        "Ticker": getattr(self.data, "_name", "UNKNOWN"),
                        "Type": order_type,
                        "Price": round(exec_price, 2),
                        "Size": order.executed.size,
                        "Value": round(order.executed.value, 2),
                        "Comm": round(order.executed.comm, 2),
                        "PnL": round(pnl, 2),
                        "RSI": round(self.rsi[0], 2),
                    }
                )

        self.order = None

    def next(self):
        """Execute buy/sell logic."""
        if self.position:
            rsi_overbought = self.rsi[0] > self.params.rsi_overbought
            below_exit_ma = self.data.close[0] < self.exit_ma[0]

            if rsi_overbought or below_exit_ma:
                self.order = self.sell()
        else:
            if self.order:
                return

            in_uptrend = self.data.close[0] > self.ma[0]
            rsi_oversold = self.rsi[0] < self.params.rsi_threshold

            if in_uptrend and rsi_oversold:
                self.order = self.buy()


class BuyAndHoldStrategy(bt.Strategy):
    """Benchmark: Buy on day 1, hold until end of backtest."""

    def __init__(self):
        self.ordered = False
        self.trade_logs = []

    def next(self):
        if not self.ordered:
            self.buy()
            self.ordered = True


# ==========================================
# DATA HELPERS
# ==========================================
def fetch_and_clean_data(ticker, years=10):
    """Fetch and format daily data from yfinance for Backtrader."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print(f"  Downloading {ticker}...", end=" ", flush=True)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        print("ERROR: No data")
        return None

    # Flatten MultiIndex columns (yfinance compatibility fix)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = "datetime"
    print(f"OK ({len(data)} bars)")
    return data


# ==========================================
# BACKTEST ENGINE
# ==========================================
def run_single_backtest(
    data,
    strategy_class,
    strat_params=None,
    initial_cash=10000.0,
    sizer_pct=95,
):
    """Execute a single backtest run and return performance metrics."""
    strat_params = strat_params or {}

    cerebro = bt.Cerebro()
    bt_data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(bt_data)
    cerebro.addstrategy(strategy_class, **strat_params)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    # Deploy capital: use PercentSizer for proper position sizing
    cerebro.addsizer(bt.sizers.PercentSizer, percents=sizer_pct)

    # Performance Analyzers
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        riskfreerate=0.01,
        annualize=True,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    results = cerebro.run()
    strat = results[0]

    final_val = cerebro.broker.getvalue()
    tot_return = (final_val - initial_cash) / initial_cash * 100.0

    # Safe Analyzer Extraction (handles zero-trade edge cases)
    try:
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    except Exception:
        sharpe = None

    try:
        max_dd = (
            strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0)
        )
    except Exception:
        max_dd = 0.0

    try:
        trade_info = strat.analyzers.trades.get_analysis()
        total_trades = trade_info.get("total", {}).get("closed", 0)
    except Exception:
        total_trades = 0

    return {
        "Final Value": final_val,
        "Return (%)": tot_return,
        "Sharpe": sharpe if sharpe is not None else "N/A",
        "Max DD (%)": max_dd,
        "Total Trades": total_trades,
        "Logs": getattr(strat, "trade_logs", []),
    }


# ==========================================
# FEATURE 1: BENCHMARK COMPARISON
# ==========================================
def run_benchmark_comparison(tickers, years=10):
    """Compare RSI Uptrend Strategy against Buy-and-Hold benchmark."""
    print(f"\n{'=' * 80}")
    print("FEATURE 1: STRATEGY VS. BUY-AND-HOLD BENCHMARK")
    print(f"{'=' * 80}\n")

    results = []
    all_logs = []

    for ticker in tickers:
        data = fetch_and_clean_data(ticker, years)
        if data is None:
            continue

        # Run RSI Strategy
        print(f"  Running RSI Strategy for {ticker}...", end=" ", flush=True)
        strat_res = run_single_backtest(data, RSIUptrendStrategy)
        print("Done")

        # Run Buy & Hold Benchmark
        print(f"  Running Buy & Hold for {ticker}...", end=" ", flush=True)
        bnh_res = run_single_backtest(data, BuyAndHoldStrategy, sizer_pct=95)
        print("Done")

        alpha = strat_res["Return (%)"] - bnh_res["Return (%)"]

        results.append(
            {
                "Ticker": ticker,
                "Strategy Return (%)": f"{strat_res['Return (%)']:.2f}",
                "B&H Return (%)": f"{bnh_res['Return (%)']:.2f}",
                "Alpha (%)": f"{alpha:.2f}",
                "Strat Max DD (%)": f"{strat_res['Max DD (%)']:.2f}",
                "B&H Max DD (%)": f"{bnh_res['Max DD (%)']:.2f}",
                "Strat Trades": strat_res["Total Trades"],
            }
        )

        if strat_res["Logs"]:
            all_logs.extend(strat_res["Logs"])

    print(f"\n{'-' * 80}")
    print("BENCHMARK RESULTS")
    print(f"{'-' * 80}")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Export CSV Trade Logs
    if all_logs:
        logs_df = pd.DataFrame(all_logs)
        logs_path = "trade_logs_benchmark.csv"
        logs_df.to_csv(logs_path, index=False)
        print(f"\n✓ Trade logs exported to '{logs_path}'")
        print(f"  Total trade records: {len(all_logs)}")

    print(f"\n{'=' * 80}\n")


# ==========================================
# FEATURE 2: PARAMETER SWEEP
# ==========================================
def run_parameter_sweep(ticker, years=10):
    """Sweep RSI thresholds and MA periods to evaluate robustness."""
    print(f"\n{'=' * 80}")
    print(f"FEATURE 2: PARAMETER SWEEP FOR {ticker}")
    print(f"{'=' * 80}\n")

    data = fetch_and_clean_data(ticker, years)
    if data is None:
        print(f"Error: Could not fetch data for {ticker}")
        return

    rsi_thresholds = [25, 30, 35, 40]
    ma_periods = [20, 50, 100, 200]
    sweep_results = []

    print("Testing parameter combinations...")
    total_combos = len(rsi_thresholds) * len(ma_periods)
    combo_count = 0

    for rsi_thresh in rsi_thresholds:
        for ma_p in ma_periods:
            combo_count += 1
            print(f"  [{combo_count}/{total_combos}] RSI={rsi_thresh}, MA={ma_p}...", end=" ", flush=True)

            res = run_single_backtest(
                data,
                RSIUptrendStrategy,
                strat_params={
                    "rsi_threshold": rsi_thresh,
                    "uptrend_ma_period": ma_p,
                    "log_trades": False,
                },
            )

            sweep_results.append(
                {
                    "RSI Threshold": rsi_thresh,
                    "MA Period": ma_p,
                    "Return (%)": round(res["Return (%)"], 2),
                    "Sharpe": (
                        round(res["Sharpe"], 2)
                        if isinstance(res["Sharpe"], float)
                        else "N/A"
                    ),
                    "Max DD (%)": round(res["Max DD (%)"], 2),
                    "Total Trades": res["Total Trades"],
                }
            )
            print("OK")

    print(f"\n{'-' * 80}")
    print("PARAMETER SWEEP RESULTS")
    print(f"{'-' * 80}")
    sweep_df = pd.DataFrame(sweep_results)
    print(sweep_df.to_string(index=False))

    # Find best parameters
    best_return_idx = sweep_df["Return (%)"].idxmax()
    best_params = sweep_df.iloc[best_return_idx]
    print(f"\n✓ Best parameters (by return): RSI={int(best_params['RSI Threshold'])}, "
          f"MA={int(best_params['MA Period'])} "
          f"({best_params['Return (%)']:.2f}% return)")

    # Export to CSV
    sweep_path = f"parameter_sweep_{ticker}.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"✓ Parameter sweep exported to '{sweep_path}'")

    print(f"\n{'=' * 80}\n")


# ==========================================
# FEATURE 3: BASKET BACKTEST
# ==========================================
def run_basket_backtest(tickers, years=10, initial_cash=50000.0):
    """Run multi-asset basket backtest with equal capital allocation."""
    print(f"\n{'=' * 80}")
    print(f"FEATURE 3: MULTI-TICKER BASKET BACKTEST")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"{'=' * 80}\n")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    # Equal capital allocation: e.g., 18% per stock if 5 tickers (90% deployed)
    pct_per_stock = round(90.0 / len(tickers), 1)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=pct_per_stock)

    print("Loading data for all tickers...")
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"  WARNING: No data for {ticker}, skipping")
            continue

        # Normalize columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index.name = "datetime"

        bt_data = bt.feeds.PandasData(dataname=data, name=ticker)
        cerebro.adddata(bt_data)
        print(f"  ✓ {ticker} loaded")

    # Add strategy and analyzers
    cerebro.addstrategy(RSIUptrendStrategy, log_trades=False)
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        riskfreerate=0.01,
        annualize=True,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    print("\nRunning basket backtest...")
    results = cerebro.run()
    strat = results[0]

    final_val = cerebro.broker.getvalue()
    tot_ret = (final_val - initial_cash) / initial_cash * 100.0

    # Extract metrics
    try:
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    except Exception:
        sharpe = None

    try:
        max_dd = (
            strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0)
        )
    except Exception:
        max_dd = 0.0

    try:
        trade_info = strat.analyzers.trades.get_analysis()
        total_trades = trade_info.get("total", {}).get("closed", 0)
    except Exception:
        total_trades = 0

    print(f"\n{'-' * 80}")
    print("BASKET BACKTEST RESULTS")
    print(f"{'-' * 80}")
    print(f"Initial Cash:        ${initial_cash:,.2f}")
    print(f"Final Value:         ${final_val:,.2f}")
    print(f"Total Return:        {tot_ret:.2f}%")
    print(f"Sharpe Ratio:        {sharpe if sharpe is not None else 'N/A'}")
    print(f"Max Drawdown:        {max_dd:.2f}%")
    print(f"Total Trades:        {total_trades}")
    print(f"Capital per ticker:  {pct_per_stock}% of portfolio")

    print(f"\n{'=' * 80}\n")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RSI UPTREND BACKTEST - ADVANCED ANALYSIS")
    print("=" * 80)

    tech_basket = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]

    # Feature 1: Strategy vs. Buy & Hold Benchmark + CSV Export
    run_benchmark_comparison(tech_basket, years=10)

    # Feature 2: Parameter Sweep Optimization
    run_parameter_sweep("AAPL", years=10)

    # Feature 3: Multi-Ticker Basket Testing
    run_basket_backtest(tech_basket, years=10, initial_cash=50000.0)

    print("✓ All backtests completed!")
