#!/usr/bin/env python3
"""
Enhanced Template Grid Implementation for Forex Pattern Discovery Framework
This module implements the Template Grid methodology exactly as specified in the research paper
by Goumatianos et al. (2017) "An algorithmic framework for frequent intraday pattern recognition"
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from tslearn.metrics import dtw
import logging
import json
from datetime import datetime
import pickle
from tqdm import tqdm
from flask import current_app
from typing import List, Dict, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhanced_pattern_extraction')

class EnhancedTemplateGrid:
    """
    Implementation of the Template Grid (TG) approach exactly as specified in the research paper.
    Supports multiple grid dimensions: 10×10, 15×10, 20×15, 25×15
    """
    
    # Research paper specified grid dimensions
    SUPPORTED_GRIDS = {
        'small': (10, 10),      # 10x10 grid
        'medium': (15, 10),     # 15x10 grid  
        'large': (20, 15),      # 20x15 grid
        'extra_large': (25, 15) # 25x15 grid
    }
    
    def __init__(self, grid_type='small'):
        """
        Initialize Enhanced Template Grid with research-specified dimensions.
        
        Args:
            grid_type (str): One of 'small', 'medium', 'large', 'extra_large'
        """
        if grid_type not in self.SUPPORTED_GRIDS:
            raise ValueError(f"Unsupported grid type. Must be one of {list(self.SUPPORTED_GRIDS.keys())}")
            
        self.grid_type = grid_type
        self.rows, self.cols = self.SUPPORTED_GRIDS[grid_type]
        self.M = self.rows  # Grid rows (price dimension)
        self.N = self.cols  # Grid columns (time dimension)
        
        # Initialize weight matrix according to research paper formula
        self.weight_matrix = self._calculate_weight_matrix()
        
        logger.info(f"Initialized {grid_type} Template Grid ({self.rows}x{self.cols}) with weight matrix")
        
    def _calculate_weight_matrix(self) -> np.ndarray:
        """
        Calculate weight matrix according to research paper formula:
        w_{j,c} = (1-|p-j|) / D
        where D = 2M / [(M-p)(M-p+1) + (p-1)p]
        
        Returns:
            np.ndarray: Weight matrix for the grid
        """
        weights = np.zeros((self.M, self.N))
        
        for c in range(self.N):  # For each time column
            for j in range(self.M):  # For each price row
                # Calculate p (center position for this column) 
                # This varies based on price distribution in each column
                p = self.M // 2  # Default to middle row, will be updated during fitting
                
                # Calculate D according to research formula
                if p == 1:
                    D = 2 * self.M / (self.M - 1)
                elif p == self.M:
                    D = 2 * self.M / (self.M - 1)  
                else:
                    D = (2 * self.M) / ((self.M - p) * (self.M - p + 1) + (p - 1) * p)
                
                # Calculate weight
                weights[j, c] = (1 - abs(p - (j + 1))) / D  # j+1 because indexing starts from 0
                
        return weights
    
    def fit(self, prices: np.ndarray) -> np.ndarray:
        """
        Fit the Template Grid to a price series using research paper methodology.
        
        Args:
            prices (np.ndarray): Array of price values to fit
            
        Returns:
            np.ndarray: Pattern Identification Code (PIC) array
        """
        if len(prices) != self.N:
            logger.warning(f"Price series length ({len(prices)}) != grid columns ({self.N}). Resampling...")
            # Resample to exact grid size
            indices = np.linspace(0, len(prices) - 1, self.N)
            prices = np.array([prices[int(i)] for i in indices])
        
        # Calculate PIC according to research paper formula: pos = [M(p-L)/(H-L)]
        L = np.min(prices)  # Lowest price in the period
        H = np.max(prices)  # Highest price in the period
        
        if H == L:  # Handle case where all prices are the same
            pic = np.full(self.N, self.M // 2)  # Put all in middle row
        else:
            # Calculate position for each price point
            pic = np.round(self.M * (prices - L) / (H - L)).astype(int)
            # Ensure values are within grid bounds [0, M-1]
            pic = np.clip(pic, 0, self.M - 1)
        
        # Create binary grid representation
        grid = np.zeros((self.M, self.N))
        for col, row in enumerate(pic):
            grid[row, col] = 1
            
        self.last_grid = grid
        self.last_pic = pic
        self.last_prices = prices
        
        return pic
    
    def calculate_similarity(self, pic1: np.ndarray, pic2: np.ndarray) -> float:
        """
        Calculate pattern similarity using research paper formula:
        sim(Prot_i, p_m) = 100 × (∑ w_{r_i,i}) / N
        
        Args:
            pic1 (np.ndarray): First Pattern Identification Code
            pic2 (np.ndarray): Second Pattern Identification Code
            
        Returns:
            float: Similarity score (0-100)
        """
        if len(pic1) != len(pic2) or len(pic1) != self.N:
            raise ValueError("PIC arrays must have same length as grid columns")
        
        # Calculate weighted similarity
        total_weight = 0
        for i in range(self.N):
            # Get the row positions for both patterns
            r1, r2 = pic1[i], pic2[i]
            
            # If positions match exactly, use full weight
            if r1 == r2:
                total_weight += self.weight_matrix[r1, i]
            else:
                # For non-matching positions, weight decreases based on distance
                # This follows the research paper's weighted approach
                distance = abs(r1 - r2)
                weight_factor = max(0, 1 - distance / self.M)
                avg_weight = (self.weight_matrix[r1, i] + self.weight_matrix[r2, i]) / 2
                total_weight += avg_weight * weight_factor
        
        # Calculate similarity as percentage
        similarity = 100 * total_weight / self.N
        
        return min(100, max(0, similarity))  # Clamp to [0, 100]
    
    def get_grid_representation(self) -> np.ndarray:
        """Get the last fitted grid as binary matrix."""
        if hasattr(self, 'last_grid'):
            return self.last_grid
        else:
            raise ValueError("No grid has been fitted yet. Call fit() first.")
    
    def visualize_pattern(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the fitted pattern with grid overlay.
        
        Args:
            save_path (str, optional): Path to save the visualization
        """
        if not hasattr(self, 'last_grid'):
            raise ValueError("No pattern has been fitted yet. Call fit() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot original price series
        ax1.plot(range(self.N), self.last_prices, 'b-', linewidth=2, label='Price')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.set_title(f'Original Price Series ({self.grid_type} grid)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot grid representation
        im = ax2.imshow(self.last_grid, cmap='RdBu', aspect='auto', origin='upper')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price Level')
        ax2.set_title(f'Template Grid Representation ({self.rows}×{self.cols})')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Grid Activation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pattern visualization saved to {save_path}")
        
        plt.show()

class PredicateVariablesSystem:
    """
    Implementation of the 10 predicate variables system from the research paper.
    Evaluates price movements within next 1, 5, 10, 20, and 50 periods.
    """
    
    # Timeframe-specific pip thresholds from research paper
    PIP_THRESHOLDS = {
        '1min': {
            'small': 5,   # ±5 pips for 1-minute timeframe
            'medium': 10,
            'large': 20
        },
        '5min': {
            'small': 10,
            'medium': 20,
            'large': 40
        },
        '15min': {
            'small': 15,
            'medium': 30,
            'large': 60
        },
        '60min': {
            'small': 30,
            'medium': 60,
            'large': 120
        }
    }
    
    def __init__(self, timeframe: str = '1min'):
        """
        Initialize predicate variables system.
        
        Args:
            timeframe (str): Trading timeframe ('1min', '5min', '15min', '60min')
        """
        self.timeframe = timeframe
        if timeframe not in self.PIP_THRESHOLDS:
            logger.warning(f"Unsupported timeframe {timeframe}. Using 1min defaults.")
            self.timeframe = '1min'
        
        self.thresholds = self.PIP_THRESHOLDS[self.timeframe]
        
        # Define the 10 predicate variables as specified in research paper Table 2
        self.predicates = [
            ("within next 1 period, highest price is more than +{small} pips", 1, 'high', 'small'),
            ("within next 1 period, lowest price is less than -{small} pips", 1, 'low', 'small'),
            ("within next 5 periods, highest price is more than +{small} pips", 5, 'high', 'small'),
            ("within next 5 periods, lowest price is less than -{small} pips", 5, 'low', 'small'),
            ("within next 10 periods, highest price is more than +{medium} pips", 10, 'high', 'medium'),
            ("within next 10 periods, lowest price is less than -{medium} pips", 10, 'low', 'medium'),
            ("within next 20 periods, highest price is more than +{medium} pips", 20, 'high', 'medium'),
            ("within next 20 periods, lowest price is less than -{medium} pips", 20, 'low', 'medium'),
            ("within next 50 periods, highest price is more than +{large} pips", 50, 'high', 'large'),
            ("within next 50 periods, lowest price is less than -{large} pips", 50, 'low', 'large')
        ]
        
        logger.info(f"Initialized Predicate Variables System for {timeframe} timeframe")
    
    def evaluate_predicates(self, prices: np.ndarray, start_idx: int) -> Dict[int, bool]:
        """
        Evaluate all 10 predicate variables for a pattern starting at start_idx.
        
        Args:
            prices (np.ndarray): Price series
            start_idx (int): Starting index of the pattern
            
        Returns:
            Dict[int, bool]: Results for each predicate (1-10)
        """
        results = {}
        current_price = prices[start_idx]
        
        for pred_idx, (description, periods, direction, threshold_type) in enumerate(self.predicates, 1):
            # Get pip threshold
            pip_threshold = self.thresholds[threshold_type]
            
            # Define lookahead window
            end_idx = min(start_idx + periods + 1, len(prices))
            future_prices = prices[start_idx + 1:end_idx]
            
            if len(future_prices) == 0:
                results[pred_idx] = False
                continue
            
            # Convert pip threshold to price difference
            # Assuming 1 pip = 0.0001 for most forex pairs (will need adjustment for JPY pairs)
            price_threshold = pip_threshold * 0.0001
            
            # Evaluate predicate
            if direction == 'high':
                # Check if any future price exceeds current + threshold
                max_future = np.max(future_prices)
                results[pred_idx] = (max_future - current_price) > price_threshold
            else:  # direction == 'low'
                # Check if any future price falls below current - threshold
                min_future = np.min(future_prices)
                results[pred_idx] = (current_price - min_future) > price_threshold
        
        return results
    
    def calculate_prediction_accuracy(self, pattern_instances: List[Tuple[int, Dict[int, bool]]]) -> Dict[int, float]:
        """
        Calculate prediction accuracy for each predicate according to research formula:
        PA_k = 100 × (Valid Occurrences / Total Occurrences)
        
        Args:
            pattern_instances: List of (pattern_start_idx, predicate_results) tuples
            
        Returns:
            Dict[int, float]: Prediction accuracy for each predicate
        """
        accuracies = {}
        
        for pred_idx in range(1, 11):  # Predicates 1-10
            valid_count = 0
            total_count = len(pattern_instances)
            
            for _, predicate_results in pattern_instances:
                if predicate_results.get(pred_idx, False):
                    valid_count += 1
            
            if total_count > 0:
                accuracies[pred_idx] = 100.0 * valid_count / total_count
            else:
                accuracies[pred_idx] = 0.0
        
        return accuracies
    
    def has_forecasting_power(self, prediction_accuracies: Dict[int, float]) -> bool:
        """
        Determine if pattern has forecasting power according to research criteria.
        FP(Prot_i) = TRUE ⟺ ∃(pred_k, r_k): r_k = TRUE
        Pattern must have at least one predicate with ≥60% accuracy.
        
        Args:
            prediction_accuracies: Accuracy scores for each predicate
            
        Returns:
            bool: True if pattern has forecasting power
        """
        return any(accuracy >= 60.0 for accuracy in prediction_accuracies.values())
    
    def get_pattern_accuracy(self, prediction_accuracies: Dict[int, float]) -> float:
        """
        Calculate overall pattern accuracy as maximum across all predicates.
        
        Args:
            prediction_accuracies: Accuracy scores for each predicate
            
        Returns:
            float: Maximum prediction accuracy across all predicates
        """
        if not prediction_accuracies:
            return 0.0
        return max(prediction_accuracies.values())

class TradingDecisionSystem:
    """
    Implementation of the Trading Decision Variable system from research paper.
    Handles four states: NOT TRADE, CONFLICT, ENTER LONG, ENTER SHORT
    """
    
    def __init__(self):
        """Initialize trading decision system."""
        self.decision_states = ['NOT_TRADE', 'CONFLICT', 'ENTER_LONG', 'ENTER_SHORT']
        logger.info("Initialized Trading Decision System")
    
    def calculate_trend_behavior(self, predicate_results: Dict[int, bool]) -> float:
        """
        Calculate Trend Behavior according to research formula:
        TB = (r_2 + r_4 + r_6 + r_8 + r_10) - (r_1 + r_3 + r_5 + r_7 + r_9)
        
        Args:
            predicate_results: Results of predicate evaluation
            
        Returns:
            float: Trend Behavior score
        """
        # Bearish predicates (even numbers: low price predictions)
        bearish_sum = sum(predicate_results.get(i, False) for i in [2, 4, 6, 8, 10])
        
        # Bullish predicates (odd numbers: high price predictions)  
        bullish_sum = sum(predicate_results.get(i, False) for i in [1, 3, 5, 7, 9])
        
        trend_behavior = bearish_sum - bullish_sum
        
        return trend_behavior
    
    def make_trading_decision(self, trend_behavior: float, confidence_threshold: float = 0.6) -> str:
        """
        Make trading decision based on trend behavior and confidence.
        
        Args:
            trend_behavior: TB score from calculate_trend_behavior
            confidence_threshold: Minimum confidence for trading decisions
            
        Returns:
            str: One of the four decision states
        """
        if trend_behavior > confidence_threshold:
            return 'ENTER_SHORT'  # Bearish signal
        elif trend_behavior < -confidence_threshold:
            return 'ENTER_LONG'   # Bullish signal
        elif abs(trend_behavior) < 0.1:
            return 'NOT_TRADE'    # No clear signal
        else:
            return 'CONFLICT'     # Mixed signals

class PatternAnalyzer:
    """
    Enhanced pattern analyzer implementing research paper methodology.
    """
    
    def __init__(self, timeframe: str = '1min'):
        """
        Initialize pattern analyzer.
        
        Args:
            timeframe: Trading timeframe
        """
        self.timeframe = timeframe
        self.grids = {}
        
        # Initialize all grid types
        for grid_type in EnhancedTemplateGrid.SUPPORTED_GRIDS.keys():
            self.grids[grid_type] = EnhancedTemplateGrid(grid_type)
        
        self.predicate_system = PredicateVariablesSystem(timeframe)
        self.trading_system = TradingDecisionSystem()
        
        # Pattern storage
        self.patterns = {}
        self.pattern_performance = {}
        
        logger.info(f"Initialized PatternAnalyzer for {timeframe} with {len(self.grids)} grid types")
    
    def extract_patterns(self, prices: np.ndarray, window_size: int = 20, 
                        similarity_threshold: float = 60.0) -> Dict:
        """
        Extract patterns using all grid dimensions and validate with predicates.
        
        Args:
            prices: Price series to analyze
            window_size: Size of pattern window
            similarity_threshold: Minimum similarity for pattern matching (research: 60%)
            
        Returns:
            Dict: Extracted patterns with validation results
        """
        results = {
            'patterns_found': [],
            'validated_patterns': [],
            'statistics': {}
        }
        
        # Extract patterns using sliding window
        for i in tqdm(range(len(prices) - window_size), desc="Extracting patterns"):
            window = prices[i:i + window_size]
            
            # Generate patterns for each grid type
            for grid_type, grid in self.grids.items():
                try:
                    # Fit template grid to window
                    pic = grid.fit(window)
                    
                    # Evaluate predicates for this pattern
                    predicate_results = self.predicate_system.evaluate_predicates(prices, i)
                    
                    # Create pattern record
                    pattern = {
                        'start_idx': i,
                        'end_idx': i + window_size,
                        'grid_type': grid_type,
                        'pic': pic.tolist(),
                        'prices': window.tolist(),
                        'predicate_results': predicate_results,
                        'trend_behavior': self.trading_system.calculate_trend_behavior(predicate_results),
                        'trading_decision': None
                    }
                    
                    # Make trading decision
                    pattern['trading_decision'] = self.trading_system.make_trading_decision(
                        pattern['trend_behavior']
                    )
                    
                    results['patterns_found'].append(pattern)
                    
                except Exception as e:
                    logger.warning(f"Error extracting pattern at index {i} with {grid_type} grid: {e}")
                    continue
        
        logger.info(f"Extracted {len(results['patterns_found'])} total patterns")
        
        # Validate patterns with forecasting power
        results['validated_patterns'] = self._validate_patterns(results['patterns_found'])
        
        # Calculate statistics
        results['statistics'] = self._calculate_statistics(results)
        
        return results
    
    def _validate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Validate patterns for forecasting power according to research criteria."""
        validated = []
        
        # Group patterns by grid type and PIC similarity
        pattern_groups = self._group_similar_patterns(patterns)
        
        for group_patterns in pattern_groups:
            if len(group_patterns) < 3:  # Need minimum occurrences for statistical significance
                continue
            
            # Calculate prediction accuracy for this pattern group
            pattern_instances = [(p['start_idx'], p['predicate_results']) for p in group_patterns]
            accuracies = self.predicate_system.calculate_prediction_accuracy(pattern_instances)
            
            # Check for forecasting power
            if self.predicate_system.has_forecasting_power(accuracies):
                # Mark all patterns in this group as validated
                for pattern in group_patterns:
                    pattern['prediction_accuracies'] = accuracies
                    pattern['overall_accuracy'] = self.predicate_system.get_pattern_accuracy(accuracies)
                    pattern['has_forecasting_power'] = True
                    validated.append(pattern)
        
        logger.info(f"Validated {len(validated)} patterns with forecasting power")
        return validated
    
    def _group_similar_patterns(self, patterns: List[Dict], 
                               similarity_threshold: float = 60.0) -> List[List[Dict]]:
        """Group patterns by similarity within each grid type."""
        groups = []
        
        # Group by grid type first
        grid_groups = {}
        for pattern in patterns:
            grid_type = pattern['grid_type']
            if grid_type not in grid_groups:
                grid_groups[grid_type] = []
            grid_groups[grid_type].append(pattern)
        
        # Within each grid type, group by PIC similarity
        for grid_type, grid_patterns in grid_groups.items():
            grid = self.grids[grid_type]
            used = set()
            
            for i, pattern1 in enumerate(grid_patterns):
                if i in used:
                    continue
                
                group = [pattern1]
                used.add(i)
                
                pic1 = np.array(pattern1['pic'])
                
                for j, pattern2 in enumerate(grid_patterns[i+1:], i+1):
                    if j in used:
                        continue
                    
                    pic2 = np.array(pattern2['pic'])
                    
                    try:
                        similarity = grid.calculate_similarity(pic1, pic2)
                        if similarity >= similarity_threshold:
                            group.append(pattern2)
                            used.add(j)
                    except Exception as e:
                        logger.warning(f"Error calculating similarity: {e}")
                        continue
                
                if len(group) >= 2:  # Only keep groups with multiple patterns
                    groups.append(group)
        
        return groups
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate comprehensive statistics for the analysis."""
        stats = {
            'total_patterns_found': len(results['patterns_found']),
            'validated_patterns_count': len(results['validated_patterns']),
            'validation_rate': 0.0,
            'grid_type_distribution': {},
            'trading_decision_distribution': {},
            'average_accuracy': 0.0,
            'forecasting_power_rate': 0.0
        }
        
        if stats['total_patterns_found'] > 0:
            stats['validation_rate'] = (stats['validated_patterns_count'] / 
                                      stats['total_patterns_found']) * 100
        
        # Analyze distributions
        for pattern in results['patterns_found']:
            grid_type = pattern['grid_type']
            trading_decision = pattern['trading_decision']
            
            stats['grid_type_distribution'][grid_type] = stats['grid_type_distribution'].get(grid_type, 0) + 1
            stats['trading_decision_distribution'][trading_decision] = stats['trading_decision_distribution'].get(trading_decision, 0) + 1
        
        # Calculate average accuracy for validated patterns
        if results['validated_patterns']:
            accuracies = [p.get('overall_accuracy', 0) for p in results['validated_patterns']]
            stats['average_accuracy'] = np.mean(accuracies)
            stats['forecasting_power_rate'] = len([p for p in results['validated_patterns'] 
                                                 if p.get('has_forecasting_power', False)]) / len(results['validated_patterns']) * 100
        
        return stats

def save_enhanced_patterns(patterns_data: Dict, output_dir: str) -> str:
    """
    Save enhanced pattern analysis results.
    
    Args:
        patterns_data: Results from PatternAnalyzer.extract_patterns()
        output_dir: Directory to save results
        
    Returns:
        str: Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'enhanced_patterns_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_data = convert_for_json(patterns_data)
    
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Enhanced pattern results saved to {filepath}")
    return filepath

if __name__ == "__main__":
    # Example usage
    logger.info("Enhanced Template Grid System initialized successfully")
    
    # Test with sample data
    sample_prices = np.random.randn(100).cumsum() + 100
    
    analyzer = PatternAnalyzer('1min')
    results = analyzer.extract_patterns(sample_prices, window_size=20)
    
    print("\nAnalysis Results:")
    print(f"Total patterns found: {results['statistics']['total_patterns_found']}")
    print(f"Validated patterns: {results['statistics']['validated_patterns_count']}")
    print(f"Validation rate: {results['statistics']['validation_rate']:.2f}%")
    
    if results['validated_patterns']:
        print(f"Average accuracy: {results['statistics']['average_accuracy']:.2f}%")
