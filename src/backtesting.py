"""
Backtesting Engine - Kiểm định ngược chiến lược giao dịch
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Engine để backtest các chiến lược giao dịch
    
    Tính năng:
    - Mô phỏng giao dịch historical
    - Tính toán metrics: returns, sharpe ratio, max drawdown
    - Hỗ trợ nhiều chiến lược
    - Quản lý vốn và position sizing
    """
    
    def __init__(self, initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Args:
            initial_capital: Vốn ban đầu
            commission: Phí giao dịch (0.1% = 0.001)
            slippage: Slippage (0.05% = 0.0005)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.trades = []
        self.portfolio_values = []
        self.positions = []
    
    def reset(self):
        """Reset trạng thái backtesting"""
        self.trades = []
        self.portfolio_values = []
        self.positions = []
    
    def run_backtest(self, data: pd.DataFrame,
                    predictions: np.ndarray,
                    strategy: str = 'long_only',
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> Dict:
        """
        Chạy backtest
        
        Args:
            data: DataFrame với price data (cần có 'close')
            predictions: Array predictions từ model
            strategy: 'long_only', 'long_short', 'threshold'
            stop_loss: Stop loss % (vd: 0.05 = 5%)
            take_profit: Take profit % (vd: 0.10 = 10%)
        
        Returns:
            Dictionary với kết quả backtest
        """
        self.reset()
        
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        
        # Align data and predictions
        min_len = min(len(data), len(predictions))
        data = data.iloc[:min_len].copy()
        predictions = predictions[:min_len]
        
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            
            # Generate signal từ prediction
            if strategy == 'long_only':
                signal = self._long_only_signal(i, predictions, data)
            elif strategy == 'long_short':
                signal = self._long_short_signal(i, predictions, data)
            elif strategy == 'threshold':
                signal = self._threshold_signal(i, predictions, data)
            else:
                signal = 0
            
            # Check stop loss / take profit
            if position != 0 and entry_price > 0:
                price_change = (current_price - entry_price) / entry_price
                
                if position == 1:  # Long position
                    if stop_loss and price_change <= -stop_loss:
                        signal = -1  # Close position (hit stop loss)
                    elif take_profit and price_change >= take_profit:
                        signal = -1  # Close position (hit take profit)
                
                elif position == -1:  # Short position
                    if stop_loss and price_change >= stop_loss:
                        signal = 1  # Close position
                    elif take_profit and price_change <= -take_profit:
                        signal = 1  # Close position
            
            # Execute trade
            if signal != 0 and signal != position:
                # Close existing position
                if position != 0:
                    pnl = self._calculate_pnl(entry_price, current_price, 
                                             position, capital)
                    capital += pnl
                    
                    self.trades.append({
                        'date': data.index[i] if isinstance(data.index, pd.DatetimeIndex) else i,
                        'action': 'CLOSE',
                        'position_type': 'LONG' if position == 1 else 'SHORT',
                        'price': current_price,
                        'pnl': pnl,
                        'capital': capital
                    })
                    
                    position = 0
                
                # Open new position
                if signal != 0:
                    position = signal
                    entry_price = current_price * (1 + self.slippage * signal)
                    
                    self.trades.append({
                        'date': data.index[i] if isinstance(data.index, pd.DatetimeIndex) else i,
                        'action': 'OPEN',
                        'position_type': 'LONG' if signal == 1 else 'SHORT',
                        'price': entry_price,
                        'pnl': 0,
                        'capital': capital
                    })
            
            # Record portfolio value
            portfolio_value = capital
            if position != 0:
                unrealized_pnl = self._calculate_pnl(entry_price, current_price,
                                                     position, capital)
                portfolio_value += unrealized_pnl
            
            self.portfolio_values.append(portfolio_value)
            self.positions.append(position)
        
        # Close any remaining position
        if position != 0:
            final_price = data['Close'].iloc[-1]
            pnl = self._calculate_pnl(entry_price, final_price, position, capital)
            capital += pnl
            
            self.trades.append({
                'date': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else len(data)-1,
                'action': 'CLOSE',
                'position_type': 'LONG' if position == 1 else 'SHORT',
                'price': final_price,
                'pnl': pnl,
                'capital': capital
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(data)
        
        logger.info(f"Backtest completed: {len(self.trades)} trades, "
                   f"Final capital: ${capital:,.2f}")
        
        return metrics
    
    def _long_only_signal(self, i: int, predictions: np.ndarray, 
                         data: pd.DataFrame) -> int:
        """
        Long-only strategy: Mua khi predict tăng, bán khi predict giảm
        
        Returns: 1 (long), 0 (no position)
        """
        if i == 0:
            return 0
        
        current_price = data['Close'].iloc[i]
        predicted_price = predictions[i]
        
        if predicted_price > current_price * 1.01:  # Predict tăng > 1%
            return 1
        else:
            return 0
    
    def _long_short_signal(self, i: int, predictions: np.ndarray,
                          data: pd.DataFrame) -> int:
        """
        Long-short strategy: Long khi predict tăng, Short khi predict giảm
        
        Returns: 1 (long), -1 (short), 0 (no position)
        """
        if i == 0:
            return 0
        
        current_price = data['Close'].iloc[i]
        predicted_price = predictions[i]
        
        threshold = 0.01  # 1%
        
        if predicted_price > current_price * (1 + threshold):
            return 1  # Long
        elif predicted_price < current_price * (1 - threshold):
            return -1  # Short
        else:
            return 0  # No position
    
    def _threshold_signal(self, i: int, predictions: np.ndarray,
                         data: pd.DataFrame) -> int:
        """
        Threshold strategy: Chỉ trade khi prediction vượt ngưỡng
        """
        if i < 5:
            return 0
        
        # So sánh prediction với MA
        ma = data['Close'].iloc[i-5:i].mean()
        predicted_price = predictions[i]
        
        if predicted_price > ma * 1.02:
            return 1
        elif predicted_price < ma * 0.98:
            return -1
        else:
            return 0
    
    def _calculate_pnl(self, entry_price: float, exit_price: float,
                      position: int, capital: float) -> float:
        """
        Tính P&L cho một trade
        
        Args:
            entry_price: Giá vào lệnh
            exit_price: Giá thoát lệnh
            position: 1 (long) hoặc -1 (short)
            capital: Vốn hiện tại
        """
        # Số cổ phiếu có thể mua
        shares = capital / entry_price
        
        # P&L before costs
        if position == 1:  # Long
            pnl = shares * (exit_price - entry_price)
        else:  # Short
            pnl = shares * (entry_price - exit_price)
        
        # Subtract costs
        entry_cost = capital * self.commission
        exit_cost = (capital + pnl) * self.commission
        
        pnl = pnl - entry_cost - exit_cost
        
        return pnl
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Tính toán các metrics"""
        
        # Total return
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Returns series
        portfolio_series = pd.Series(self.portfolio_values)
        returns = portfolio_series.pct_change().dropna()
        
        # Sharpe Ratio (annualized, assuming daily data)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        # Max Drawdown
        cummax = portfolio_series.cummax()
        drawdown = (portfolio_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in self.trades if t['action'] == 'CLOSE' and t['pnl'] > 0]
        total_closed_trades = [t for t in self.trades if t['action'] == 'CLOSE']
        win_rate = len(winning_trades) / len(total_closed_trades) if total_closed_trades else 0
        
        # Average P&L
        pnls = [t['pnl'] for t in self.trades if t['action'] == 'CLOSE']
        avg_pnl = np.mean(pnls) if pnls else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(total_closed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_pnl': avg_pnl
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Lấy DataFrame của tất cả trades"""
        return pd.DataFrame(self.trades)
    
    def plot_results(self, data: pd.DataFrame):
        """Vẽ biểu đồ kết quả backtest"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and positions
        ax1 = axes[0]
        ax1.plot(data['Close'].values, label='Price', linewidth=1)
        
        # Mark entry/exit points
        trades_df = self.get_trades_df()
        if not trades_df.empty:
            opens = trades_df[trades_df['action'] == 'OPEN']
            closes = trades_df[trades_df['action'] == 'CLOSE']
            
            for _, trade in opens.iterrows():
                idx = trade.get('date', 0)
                if isinstance(idx, (pd.Timestamp, datetime)):
                    idx = data.index.get_loc(idx)
                color = 'g' if trade['position_type'] == 'LONG' else 'r'
                ax1.scatter(idx, trade['price'], marker='^', color=color, s=100, zorder=5)
            
            for _, trade in closes.iterrows():
                idx = trade.get('date', 0)
                if isinstance(idx, (pd.Timestamp, datetime)):
                    idx = data.index.get_loc(idx)
                ax1.scatter(idx, trade['price'], marker='v', color='black', s=100, zorder=5)
        
        ax1.set_title('Price and Trading Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Portfolio value
        ax2 = axes[1]
        ax2.plot(self.portfolio_values, label='Portfolio Value', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='r', linestyle='--', 
                   label='Initial Capital')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Value ($)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Drawdown
        ax3 = axes[2]
        portfolio_series = pd.Series(self.portfolio_values)
        cummax = portfolio_series.cummax()
        drawdown = (portfolio_series - cummax) / cummax * 100
        ax3.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown %')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Time')
        ax3.grid(True)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Test backtesting engine
    print("Backtesting Engine for Stock Trading Strategies")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=250, freq='D')
    prices = 100 + np.cumsum(np.random.randn(250) * 2)
    data = pd.DataFrame({'close': prices}, index=dates)
    
    # Create sample predictions (slightly ahead of actual)
    predictions = prices * (1 + np.random.randn(250) * 0.02)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    
    results = engine.run_backtest(
        data=data,
        predictions=predictions,
        strategy='long_only',
        stop_loss=0.05,
        take_profit=0.10
    )
    
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate_pct']:.2f}%")
