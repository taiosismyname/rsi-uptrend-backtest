"""
RSI Oversold + Uptrend Backtest Framework
==========================================
Buy when 14-day RSI drops below 30 during an uptrend (price > 50-day MA).
Sell when RSI reaches 70 or price closes below 20-day MA.

Critical fixes applied:
- Flattened yfinance MultiIndex columns
- Used PercentSizer for proper position sizing (95% of capital)
- Fixed PnL calculation with order.executed.pnl
- Safely handled Sharpe Ratio crashes on zero trades
- Aligned datetime index naming (yfinance 'Date' → 'datetime')
"""

from datetime import datetime, timedelta

import backtrader as bt
import pandas as pd
import yfinance as yf


class RSIUptrendStrategy(bt.Strategy):
    """Buy when 14-day RSI falls below 30 while price is above the 50-day MA."""

    params = (
        ("rsi_period", 14),
        ("rsi_threshold", 30),
        ("uptrend_ma_period", 50),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.uptrend_ma_period
        )

        self.order = None
        self.trade_count = 0

    def notify_order(self, order):
        """Handle order execution notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.trade_count += 1
                self.log(f"BUY executed at ${order.executed.price:.2f}")
            elif order.issell():
                # Use Backtrader's built-in PnL calculation (includes commissions)
                pnl = order.executed.pnl
                self.log(
                    f"SELL executed at ${order.executed.price:.2f}, P/L: ${pnl:.2f}"
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def next(self):
        """Execute strategy logic on each bar."""
        if self.position or self.order:
            return

        in_uptrend = self.data.close[0] > self.ma[0]
        rsi_oversold = self.rsi[0] < self.params.rsi_threshold

        if in_uptrend and rsi_oversold:
            self.order = self.buy()
            self.log(
                f"BUY signal - RSI: {self.rsi[0]:.2f}, "
                f"Price: ${self.data.close[0]:.2f}, "
                f"MA50: ${self.ma[0]:.2f}"
            )

    def log(self, txt, dt=None):
        """Print log message with timestamp."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")


class SimpleExitStrategy(RSIUptrendStrategy):
    """Buy on RSI oversold + uptrend, sell on RSI overbought or weak trend."""

    params = (
        ("rsi_overbought", 70),
        ("exit_ma_period", 20),
    )

    def __init__(self):
        super().__init__()
        self.exit_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.exit_ma_period
        )

    def next(self):
        """Execute strategy with explicit exit conditions."""
        if self.position:
            rsi_overbought = self.rsi[0] > self.params.rsi_overbought
            below_exit_ma = self.data.close[0] < self.exit_ma[0]

            if rsi_overbought or below_exit_ma:
                self.order = self.sell()
                self.log(
                    f"SELL signal - RSI: {self.rsi[0]:.2f}, "
                    f"Price: ${self.data.close[0]:.2f}"
                )
        else:
            if self.order:
                return

            in_uptrend = self.data.close[0] > self.ma[0]
            rsi_oversold = self.rsi[0] < self.params.rsi_threshold

            if in_uptrend and rsi_oversold:
                self.order = self.buy()
                self.log(
                    f"BUY signal - RSI: {self.rsi[0]:.2f}, "
                    f"Price: ${self.data.close[0]:.2f}, "
                    f"MA50: ${self.ma[0]:.2f}"
                )


def normalize_yfinance_data(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns and ensure clean datetime index."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index.name = "datetime"
    return data


def run_backtest(ticker, strategy_class=SimpleExitStrategy, years=10):
    """
    Run backtest for a given ticker.

    Args:
        ticker (str): Stock ticker symbol
        strategy_class: Strategy class to use
        years (int): Number of years to backtest

    Returns:
        tuple: (cerebro instance, strategy instance) or (None, None) on error
    """
    print(f"\n{'=' * 60}")
    print(f"Running backtest for {ticker} ({years} years)")
    print(f"{'=' * 60}\n")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    print(f"Downloading data from {start_date.date()} to {end_date.date()}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        print(f"Error: No data found for {ticker}")
        return None, None

    data = normalize_yfinance_data(data)
    print(f"Downloaded {len(data)} bars\n")

    cerebro = bt.Cerebro()

    # Create Backtrader feed with normalized data
    bt_data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(bt_data)
    cerebro.addstrategy(strategy_class)

    initial_cash = 10000.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    # Use 95% of total portfolio value per trade
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # Performance Analyzers
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        riskfreerate=0.01,
        annualize=True,
    )
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    print("Running backtest...\n")
    results = cerebro.run()
    strategy = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {strategy.trade_count}")

    # Safe extraction of analyzer values (handles zero-trade edge cases)
    try:
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        sharpe_val = sharpe_analysis.get("sharperatio", None)
        if sharpe_val is not None:
            print(f"Sharpe Ratio: {sharpe_val:.2f}")
        else:
            print("Sharpe Ratio: N/A (Insufficient trade activity)")
    except Exception:
        print("Sharpe Ratio: N/A")

    try:
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        max_dd = drawdown_analysis.get("max", {}).get("drawdown", 0.0)
        print(f"Max Drawdown: {max_dd:.2f}%")
    except Exception:
        pass

    print(f"{'=' * 60}\n")

    return cerebro, strategy


def main():
    """Run backtests across all tech stocks and print summary."""
    tech_stocks = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    results_summary = []

    for ticker in tech_stocks:
        try:
            cerebro, strategy = run_backtest(ticker, SimpleExitStrategy, years=10)
            if cerebro:
                final_value = cerebro.broker.getvalue()
                total_return = (final_value - 10000) / 10000 * 100
                results_summary.append(
                    {
                        "Ticker": ticker,
                        "Final Value": f"${final_value:,.2f}",
                        "Total Return": f"{total_return:.2f}%",
                        "Trades": strategy.trade_count,
                    }
                )
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}\n")

    if results_summary:
        print(f"\n{'=' * 60}")
        print("SUMMARY OF ALL STOCKS")
        print(f"{'=' * 60}")
        df = pd.DataFrame(results_summary)
        print(df.to_string(index=False))
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
