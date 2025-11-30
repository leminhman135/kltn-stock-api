"""
Backtesting Engine - Đánh giá hiệu quả chiến lược giao dịch
Bao gồm: Sharpe Ratio, Max Drawdown, Win Rate, Portfolio Simulation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Loại tín hiệu giao dịch"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PositionType(Enum):
    """Loại vị thế"""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Trade:
    """Thông tin một giao dịch"""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    position_type: PositionType = PositionType.LONG
    quantity: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None
    
    @property
    def pnl(self) -> float:
        """Profit/Loss của giao dịch"""
        if not self.is_closed:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        """Profit/Loss theo phần trăm"""
        if not self.is_closed:
            return 0.0
        return (self.pnl / (self.entry_price * self.quantity)) * 100
    
    @property
    def holding_days(self) -> int:
        """Số ngày nắm giữ"""
        if not self.is_closed:
            return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    """Kết quả backtest"""
    # Basic Info
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Performance Metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk Metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    
    # Detailed Data
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Chuyển kết quả sang dictionary"""
        return {
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_capital': self.initial_capital,
            'final_capital': round(self.final_capital, 2),
            'total_return': round(self.total_return, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'annualized_return': round(self.annualized_return, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'sortino_ratio': round(self.sortino_ratio, 3),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'volatility': round(self.volatility, 2),
            'var_95': round(self.var_95, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_holding_days': round(self.avg_holding_days, 1),
        }


class BacktestingEngine:
    """
    Engine backtest chiến lược giao dịch
    
    Tính năng:
    - Mô phỏng giao dịch theo tín hiệu
    - Tính toán các metrics: Sharpe, Sortino, Max Drawdown, Win Rate
    - Hỗ trợ Stop Loss, Take Profit
    - Portfolio tracking
    
    Ví dụ sử dụng:
    ```python
    engine = BacktestingEngine(initial_capital=100_000_000)
    result = engine.run(price_data, signals)
    print(result.sharpe_ratio)
    ```
    """
    
    def __init__(self, 
                 initial_capital: float = 100_000_000,  # 100 triệu VND
                 commission_rate: float = 0.001,  # 0.1% phí giao dịch
                 slippage: float = 0.001,  # 0.1% slippage
                 risk_free_rate: float = 0.05):  # 5% lãi suất phi rủi ro
        """
        Args:
            initial_capital: Vốn ban đầu (VND)
            commission_rate: Phí giao dịch (%)
            slippage: Slippage (%)
            risk_free_rate: Lãi suất phi rủi ro hàng năm (cho Sharpe Ratio)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        # State
        self.capital = initial_capital
        self.position: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
    
    def reset(self):
        """Reset engine về trạng thái ban đầu"""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Áp dụng slippage vào giá"""
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def _calculate_commission(self, value: float) -> float:
        """Tính phí giao dịch"""
        return value * self.commission_rate
    
    def _open_position(self, symbol: str, price: float, date: datetime,
                      position_type: PositionType = PositionType.LONG,
                      stop_loss_pct: Optional[float] = None,
                      take_profit_pct: Optional[float] = None) -> bool:
        """Mở vị thế mới"""
        if self.position is not None:
            return False
        
        # Áp dụng slippage
        actual_price = self._apply_slippage(price, is_buy=(position_type == PositionType.LONG))
        
        # Tính số lượng có thể mua
        commission = self._calculate_commission(self.capital)
        available = self.capital - commission
        quantity = int(available / actual_price)
        
        if quantity <= 0:
            return False
        
        # Tính stop loss và take profit
        stop_loss = None
        take_profit = None
        
        if stop_loss_pct:
            if position_type == PositionType.LONG:
                stop_loss = actual_price * (1 - stop_loss_pct)
            else:
                stop_loss = actual_price * (1 + stop_loss_pct)
        
        if take_profit_pct:
            if position_type == PositionType.LONG:
                take_profit = actual_price * (1 + take_profit_pct)
            else:
                take_profit = actual_price * (1 - take_profit_pct)
        
        # Tạo trade
        self.position = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=actual_price,
            position_type=position_type,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Trừ vốn
        self.capital -= (actual_price * quantity + commission)
        
        logger.debug(f"Opened {position_type.value} position: {quantity} @ {actual_price:.0f}")
        return True
    
    def _close_position(self, price: float, date: datetime) -> Optional[Trade]:
        """Đóng vị thế hiện tại"""
        if self.position is None:
            return None
        
        # Áp dụng slippage
        is_buy = (self.position.position_type == PositionType.SHORT)
        actual_price = self._apply_slippage(price, is_buy=is_buy)
        
        # Cập nhật trade
        self.position.exit_date = date
        self.position.exit_price = actual_price
        
        # Tính tiền nhận được
        value = actual_price * self.position.quantity
        commission = self._calculate_commission(value)
        self.capital += (value - commission)
        
        # Lưu trade
        closed_trade = self.position
        self.trades.append(closed_trade)
        self.position = None
        
        logger.debug(f"Closed position: {closed_trade.pnl_percent:.2f}% PnL")
        return closed_trade
    
    def _check_stop_loss_take_profit(self, high: float, low: float, 
                                     date: datetime) -> bool:
        """Kiểm tra stop loss và take profit"""
        if self.position is None:
            return False
        
        # Check stop loss
        if self.position.stop_loss:
            if self.position.position_type == PositionType.LONG:
                if low <= self.position.stop_loss:
                    self._close_position(self.position.stop_loss, date)
                    return True
            else:  # SHORT
                if high >= self.position.stop_loss:
                    self._close_position(self.position.stop_loss, date)
                    return True
        
        # Check take profit
        if self.position.take_profit:
            if self.position.position_type == PositionType.LONG:
                if high >= self.position.take_profit:
                    self._close_position(self.position.take_profit, date)
                    return True
            else:  # SHORT
                if low <= self.position.take_profit:
                    self._close_position(self.position.take_profit, date)
                    return True
        
        return False
    
    def _get_portfolio_value(self, current_price: float) -> float:
        """Tính tổng giá trị danh mục"""
        value = self.capital
        if self.position:
            value += self.position.quantity * current_price
        return value
    
    def run(self, 
            price_data: pd.DataFrame,
            signals: pd.Series,
            symbol: str = "STOCK",
            stop_loss_pct: Optional[float] = 0.05,  # 5% stop loss
            take_profit_pct: Optional[float] = 0.10,  # 10% take profit
            ) -> BacktestResult:
        """
        Chạy backtest
        
        Args:
            price_data: DataFrame với cột ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            signals: Series với index là date, giá trị là SignalType ('BUY', 'SELL', 'HOLD')
            symbol: Mã cổ phiếu
            stop_loss_pct: % stop loss (None = không dùng)
            take_profit_pct: % take profit (None = không dùng)
        
        Returns:
            BacktestResult với đầy đủ metrics
        """
        self.reset()
        
        # Validate input
        if price_data.empty:
            raise ValueError("price_data is empty")
        
        # Ensure date column
        if 'date' not in price_data.columns:
            if isinstance(price_data.index, pd.DatetimeIndex):
                price_data = price_data.reset_index()
                price_data.rename(columns={'index': 'date'}, inplace=True)
            else:
                raise ValueError("price_data must have 'date' column or DatetimeIndex")
        
        price_data['date'] = pd.to_datetime(price_data['date'])
        price_data = price_data.sort_values('date').reset_index(drop=True)
        
        # Initialize
        start_date = price_data['date'].iloc[0]
        end_date = price_data['date'].iloc[-1]
        prev_value = self.initial_capital
        
        logger.info(f"Starting backtest for {symbol}: {start_date.date()} to {end_date.date()}")
        
        # Main loop
        for idx, row in price_data.iterrows():
            date = row['date']
            open_price = row['Open']
            high = row['High']
            low = row['Low']
            close = row['Close']
            
            # Check stop loss / take profit
            self._check_stop_loss_take_profit(high, low, date)
            
            # Get signal for this date
            signal = signals.get(date, SignalType.HOLD)
            if isinstance(signal, str):
                try:
                    signal = SignalType(signal.upper())
                except ValueError:
                    signal = SignalType.HOLD
            
            # Execute signal
            if signal == SignalType.BUY and self.position is None:
                self._open_position(
                    symbol, open_price, date,
                    PositionType.LONG,
                    stop_loss_pct, take_profit_pct
                )
            elif signal == SignalType.SELL and self.position is not None:
                self._close_position(open_price, date)
            
            # Track equity
            current_value = self._get_portfolio_value(close)
            self.equity_curve.append(current_value)
            
            # Track daily returns
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            self.daily_returns.append(daily_return)
            prev_value = current_value
        
        # Close any remaining position
        if self.position is not None:
            last_price = price_data['Close'].iloc[-1]
            self._close_position(last_price, end_date)
        
        # Calculate metrics
        result = self._calculate_metrics(symbol, start_date, end_date)
        
        logger.info(f"Backtest complete: {result.total_return_pct:.2f}% return, "
                   f"Sharpe: {result.sharpe_ratio:.2f}, Max DD: {result.max_drawdown_pct:.2f}%")
        
        return result
    
    def _calculate_metrics(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> BacktestResult:
        """Tính toán các metrics từ kết quả backtest"""
        
        # Basic returns
        final_capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25 if days > 0 else 1
        annualized_return = ((final_capital / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Daily returns array
        returns_arr = np.array(self.daily_returns)
        
        # Volatility (annualized)
        volatility = np.std(returns_arr) * np.sqrt(252) * 100 if len(returns_arr) > 1 else 0
        
        # Sharpe Ratio
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns_arr - daily_rf
        sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0
        
        # Sortino Ratio (only downside volatility)
        downside_returns = returns_arr[returns_arr < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (np.mean(excess_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # Max Drawdown
        equity_arr = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_drawdown_pct = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
        max_drawdown = np.max(peak - equity_arr) if len(drawdown) > 0 else 0
        
        # VaR 95%
        var_95 = np.percentile(returns_arr, 5) * final_capital if len(returns_arr) > 0 else 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average holding days
        avg_holding = np.mean([t.holding_days for t in self.trades]) if self.trades else 0
        
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            volatility=volatility,
            var_95=var_95,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding,
            trades=self.trades,
            equity_curve=self.equity_curve,
            daily_returns=self.daily_returns
        )


class SignalGenerator:
    """
    Tạo tín hiệu giao dịch từ predictions
    
    Strategies:
    1. Simple: BUY khi predict > current, SELL ngược lại
    2. Threshold: BUY khi predict tăng > X%, SELL khi giảm > Y%
    3. Moving Average: BUY khi price cross up MA, SELL cross down
    4. RSI: BUY khi RSI < 30, SELL khi RSI > 70
    """
    
    @staticmethod
    def from_predictions(prices: pd.Series, predictions: pd.Series,
                        threshold: float = 0.02) -> pd.Series:
        """
        Tạo signals từ predictions
        
        Args:
            prices: Series giá hiện tại
            predictions: Series giá dự đoán cho ngày mai
            threshold: Ngưỡng % để tạo signal (mặc định 2%)
        
        Returns:
            Series với signals (BUY/SELL/HOLD)
        """
        signals = {}
        
        for date in prices.index:
            if date not in predictions.index:
                signals[date] = SignalType.HOLD.value
                continue
            
            current = prices[date]
            predicted = predictions[date]
            
            change_pct = (predicted - current) / current
            
            if change_pct > threshold:
                signals[date] = SignalType.BUY.value
            elif change_pct < -threshold:
                signals[date] = SignalType.SELL.value
            else:
                signals[date] = SignalType.HOLD.value
        
        return pd.Series(signals)
    
    @staticmethod
    def from_technical_indicators(df: pd.DataFrame) -> pd.Series:
        """
        Tạo signals từ technical indicators
        
        Args:
            df: DataFrame với các cột indicators (RSI, MACD, SMA, etc.)
        
        Returns:
            Series với signals
        """
        signals = {}
        
        for idx, row in df.iterrows():
            score = 0
            
            # RSI signal
            if 'rsi' in row:
                if row['rsi'] < 30:
                    score += 2
                elif row['rsi'] > 70:
                    score -= 2
            
            # MACD signal
            if 'macd' in row and 'macd_signal' in row:
                if row['macd'] > row['macd_signal']:
                    score += 1
                else:
                    score -= 1
            
            # SMA crossover
            if 'sma_20' in row and 'sma_50' in row:
                if row['sma_20'] > row['sma_50']:
                    score += 1
                else:
                    score -= 1
            
            # Close vs SMA
            if 'Close' in row and 'sma_20' in row:
                if row['Close'] > row['sma_20']:
                    score += 1
                else:
                    score -= 1
            
            # Convert score to signal
            if score >= 2:
                signals[idx] = SignalType.BUY.value
            elif score <= -2:
                signals[idx] = SignalType.SELL.value
            else:
                signals[idx] = SignalType.HOLD.value
        
        return pd.Series(signals)
    
    @staticmethod
    def from_sentiment(sentiment_scores: pd.Series, 
                      threshold: float = 0.3) -> pd.Series:
        """
        Tạo signals từ sentiment scores
        
        Args:
            sentiment_scores: Series với scores từ -1 đến 1
            threshold: Ngưỡng để tạo signal
        
        Returns:
            Series với signals
        """
        signals = {}
        
        for date, score in sentiment_scores.items():
            if score > threshold:
                signals[date] = SignalType.BUY.value
            elif score < -threshold:
                signals[date] = SignalType.SELL.value
            else:
                signals[date] = SignalType.HOLD.value
        
        return pd.Series(signals)


class ModelBacktester:
    """
    Backtest các mô hình dự đoán
    Wrapper để dễ dàng backtest và so sánh các models
    """
    
    def __init__(self, 
                 initial_capital: float = 100_000_000,
                 commission_rate: float = 0.001,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10):
        
        self.engine = BacktestingEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate
        )
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def backtest_model(self, 
                       model,
                       price_data: pd.DataFrame,
                       train_size: float = 0.7,
                       symbol: str = "STOCK") -> BacktestResult:
        """
        Backtest một model
        
        Args:
            model: Model có method fit() và predict()
            price_data: DataFrame với OHLCV data
            train_size: Tỷ lệ dữ liệu training
            symbol: Mã cổ phiếu
        
        Returns:
            BacktestResult
        """
        # Split data
        split_idx = int(len(price_data) * train_size)
        train_data = price_data.iloc[:split_idx]
        test_data = price_data.iloc[split_idx:]
        
        # Train model
        model.fit(train_data['Close'])
        
        # Generate predictions for test period
        predictions = {}
        for i in range(len(test_data) - 1):
            current_data = pd.concat([train_data, test_data.iloc[:i+1]])
            model.fit(current_data['Close'])
            pred = model.predict(steps=1)[0]
            
            date = test_data.iloc[i+1]['date'] if 'date' in test_data.columns else test_data.index[i+1]
            predictions[date] = pred
        
        predictions_series = pd.Series(predictions)
        
        # Generate signals
        prices = test_data.set_index('date')['Close'] if 'date' in test_data.columns else test_data['Close']
        signals = SignalGenerator.from_predictions(prices, predictions_series)
        
        # Run backtest
        result = self.engine.run(
            test_data, signals, symbol,
            self.stop_loss_pct, self.take_profit_pct
        )
        
        return result
    
    def compare_models(self, 
                       models: Dict[str, object],
                       price_data: pd.DataFrame,
                       symbol: str = "STOCK") -> pd.DataFrame:
        """
        So sánh nhiều models
        
        Args:
            models: Dictionary {model_name: model_instance}
            price_data: DataFrame với OHLCV data
            symbol: Mã cổ phiếu
        
        Returns:
            DataFrame so sánh các metrics
        """
        results = []
        
        for name, model in models.items():
            logger.info(f"Backtesting {name}...")
            try:
                result = self.backtest_model(model, price_data, symbol=symbol)
                result_dict = result.to_dict()
                result_dict['model_name'] = name
                results.append(result_dict)
            except Exception as e:
                logger.error(f"Error backtesting {name}: {str(e)}")
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('model_name')
        
        return comparison_df


# Quick backtest function for API
def quick_backtest(price_data: pd.DataFrame,
                   predictions: pd.Series,
                   symbol: str = "STOCK",
                   initial_capital: float = 100_000_000,
                   threshold: float = 0.02,
                   stop_loss_pct: float = 0.05,
                   take_profit_pct: float = 0.10) -> Dict:
    """
    Quick backtest function cho API endpoint
    
    Args:
        price_data: DataFrame với OHLCV
        predictions: Series với predicted prices
        symbol: Mã cổ phiếu
        initial_capital: Vốn ban đầu
        threshold: Ngưỡng signal (%)
        stop_loss_pct: Stop loss (%)
        take_profit_pct: Take profit (%)
    
    Returns:
        Dictionary với kết quả backtest
    """
    # Generate signals
    prices = price_data.set_index('date')['Close'] if 'date' in price_data.columns else price_data['Close']
    signals = SignalGenerator.from_predictions(prices, predictions, threshold)
    
    # Run backtest
    engine = BacktestingEngine(initial_capital=initial_capital)
    result = engine.run(price_data, signals, symbol, stop_loss_pct, take_profit_pct)
    
    return result.to_dict()


if __name__ == "__main__":
    print("Backtesting Engine for Stock Prediction")
    print("=" * 60)
    print("\nKey Metrics:")
    print("- Sharpe Ratio: Risk-adjusted return")
    print("- Sortino Ratio: Downside risk-adjusted return")
    print("- Max Drawdown: Largest peak-to-trough decline")
    print("- Win Rate: Percentage of profitable trades")
    print("- Profit Factor: Gross profit / Gross loss")
    print("\nUsage:")
    print("  engine = BacktestingEngine()")
    print("  result = engine.run(price_data, signals)")
    print("  print(result.to_dict())")
