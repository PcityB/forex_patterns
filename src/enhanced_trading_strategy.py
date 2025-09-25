#!/usr/bin/env python3
"""
Enhanced Trading Strategy Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    
class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

@dataclass
class Trade:
    direction: TradeDirection
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0

class EnhancedTradingStrategy:
    """
    Trading strategy implementing research paper methodology
    """
    
    def __init__(self, pip_value=0.0001, spread=2, initial_capital=10000):
        self.pip_value = pip_value
        self.spread_pips = spread
        self.spread = spread * pip_value
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.equity_curve = []
    
    def apply_pip_range_filter(self, pattern_data: Dict, timeframe: str) -> bool:
        """
        Apply pip range filtering as specified in research paper
        """
        pip_ranges = {
            '1min': (20, 130),
            '5min': (25, 150), 
            '15min': (30, 200),
            '60min': (30, 300)
        }
        
        if timeframe not in pip_ranges:
            return True  # No filter for unknown timeframes
        
        min_pips, max_pips = pip_ranges[timeframe]
        pattern_range = pattern_data.get('pip_range', 0)
        
        return min_pips <= pattern_range <= max_pips
    
    def calculate_position_size(self, risk_percentage=0.02) -> float:
        """
        Calculate position size based on risk management
        """
        risk_amount = self.current_capital * risk_percentage
        return risk_amount / (self.spread * 100)  # Simple position sizing
    
    def execute_trade(self, signal: Dict, current_price: float, 
                     timestamp: pd.Timestamp) -> Optional[Trade]:
        """
        Execute trade based on trading decision
        """
        decision = signal.get('trading_decision')
        
        if decision in ['NOT_TRADE', 'CONFLICT']:
            return None
        
        # Determine direction
        if decision == 'ENTER_LONG':
            direction = TradeDirection.LONG
            entry_price = current_price + self.spread/2  # Add half spread
        elif decision == 'ENTER_SHORT':
            direction = TradeDirection.SHORT  
            entry_price = current_price - self.spread/2  # Subtract half spread
        else:
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size()
        
        # Create trade
        trade = Trade(
            direction=direction,
            entry_price=entry_price,
            entry_time=timestamp,
            size=position_size
        )
        
        # Set stop loss and take profit based on pattern characteristics
        self._set_trade_levels(trade, signal)
        
        self.trades.append(trade)
        return trade
    
    def _set_trade_levels(self, trade: Trade, signal: Dict):
        """Set stop loss and take profit levels"""
        # Use pattern volatility or default levels
        atr_multiple = signal.get('atr_multiple', 2.0)
        
        if trade.direction == TradeDirection.LONG:
            trade.stop_loss = trade.entry_price - (atr_multiple * self.pip_value * 20)
            trade.take_profit = trade.entry_price + (atr_multiple * self.pip_value * 40)
        else:
            trade.stop_loss = trade.entry_price + (atr_multiple * self.pip_value * 20)  
            trade.take_profit = trade.entry_price - (atr_multiple * self.pip_value * 40)
    
    def update_trades(self, current_price: float, timestamp: pd.Timestamp):
        """Update open trades"""
        for trade in self.trades:
            if trade.status != TradeStatus.OPEN:
                continue
            
            # Check stop loss and take profit
            should_exit = False
            exit_price = current_price
            
            if trade.direction == TradeDirection.LONG:
                if current_price <= trade.stop_loss:
                    should_exit = True
                    exit_price = trade.stop_loss
                elif current_price >= trade.take_profit:
                    should_exit = True
                    exit_price = trade.take_profit
            else:  # SHORT
                if current_price >= trade.stop_loss:
                    should_exit = True
                    exit_price = trade.stop_loss
                elif current_price <= trade.take_profit:
                    should_exit = True
                    exit_price = trade.take_profit
            
            if should_exit:
                self._close_trade(trade, exit_price, timestamp)
    
    def _close_trade(self, trade: Trade, exit_price: float, timestamp: pd.Timestamp):
        """Close a trade and calculate PnL"""
        trade.exit_price = exit_price
        trade.exit_time = timestamp
        trade.status = TradeStatus.CLOSED
        
        # Calculate PnL
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.size
        
        # Subtract spread cost
        trade.pnl -= self.spread * trade.size
        
        # Update capital
        self.current_capital += trade.pnl
        
        # Record equity point
        self.equity_curve.append({
            'timestamp': timestamp,
            'capital': self.current_capital,
            'pnl': trade.pnl
        })
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return {'total_trades': 0}
        
        # Basic metrics
        total_pnl = sum(t.pnl for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / 
                           sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Calculate Sharpe ratio
        returns = [t.pnl / self.initial_capital for t in closed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_trades': len(closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.current_capital,
            'return_percentage': (self.current_capital - self.initial_capital) / self.initial_capital * 100
        }
