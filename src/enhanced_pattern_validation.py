#!/usr/bin/env python3
"""
Enhanced Pattern Validation Module implementing research paper specifications
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, mannwhitneyu
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger('enhanced_validation')

class StatisticalValidator:
    """
    Implements statistical validation according to research paper methodology
    """
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def validate_pattern_significance(self, pattern_results: List[Dict]) -> Dict:
        """
        Perform statistical significance testing on pattern results
        """
        if not pattern_results:
            return {'significant': False, 'p_value': 1.0, 't_statistic': 0.0}
        
        # Extract returns or success rates
        returns = [r.get('return', 0) for r in pattern_results]
        
        # One-sample t-test against zero (no effect)
        t_stat, p_value = ttest_1samp(returns, 0)
        
        # Determine significance
        significant = p_value < self.alpha and abs(t_stat) > 1.6  # Research threshold
        
        return {
            'significant': significant,
            'p_value': p_value,
            't_statistic': t_stat,
            'mean_return': np.mean(returns),
            'sample_size': len(returns)
        }
    
    def calculate_pattern_support(self, matched_patterns: int, total_rows: int) -> float:
        """
        Calculate pattern support: (matched patterns with forecasting power / total rows) * 100
        """
        if total_rows == 0:
            return 0.0
        return (matched_patterns / total_rows) * 100

class CrossValidationFramework:
    """
    Implements 3-fold cross-validation as specified in research paper
    """
    
    def __init__(self):
        self.n_folds = 3
        self.training_years = 2
        self.testing_years = 1
    
    def create_folds(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create 3 folds with 2 years training, 1 year testing each
        """
        data = data.sort_index()  # Ensure chronological order
        total_periods = len(data)
        
        # Calculate fold sizes
        fold_size = total_periods // self.n_folds
        
        folds = []
        for i in range(self.n_folds):
            # Define test period
            test_start = i * fold_size
            test_end = test_start + fold_size
            
            # Test data
            test_data = data.iloc[test_start:test_end]
            
            # Training data (everything except test period)
            train_data = pd.concat([
                data.iloc[:test_start],
                data.iloc[test_end:]
            ])
            
            folds.append((train_data, test_data))
        
        return folds
    
    def validate_patterns(self, patterns: List, data: pd.DataFrame, 
                         pattern_analyzer) -> Dict:
        """
        Perform 3-fold cross-validation on patterns
        """
        folds = self.create_folds(data)
        results = []
        
        for fold_idx, (train_data, test_data) in enumerate(folds):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_folds}")
            
            # Train patterns on training data
            trained_patterns = pattern_analyzer.train_patterns(patterns, train_data)
            
            # Validate on test data
            fold_results = pattern_analyzer.validate_patterns(trained_patterns, test_data)
            fold_results['fold'] = fold_idx
            results.append(fold_results)
        
        # Aggregate results across folds
        return self._aggregate_cv_results(results)
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results"""
        accuracies = [r['accuracy'] for r in fold_results]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'fold_results': fold_results,
            'cv_score': np.mean(accuracies)
        }
