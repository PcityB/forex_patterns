"""
Pattern Analysis Module for Forex Pattern Discovery Framework
This module implements machine learning and statistical analysis methods for pattern identification and validation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu
import json
import pickle
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pattern_analysis')


def make_json_serializable(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        val = float(obj)
        if np.isinf(val):
            return "inf" if val > 0 else "-inf"
        return val
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float):
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    else:
        return obj

def restore_json_infinity(obj):
    """Convert JSON 'inf' and '-inf' strings back to float infinity"""
    if isinstance(obj, dict):
        return {k: restore_json_infinity(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [restore_json_infinity(i) for i in obj]
    elif obj == "inf":
        return float('inf')
    elif obj == "-inf":
        return -float('inf')
    return obj

class PatternAnalyzer:
    """
    Class for analyzing and validating extracted patterns using machine learning and statistical methods.
    """
    
    def __init__(self, patterns_dir, data_dir, output_dir=None):
        """
        Initialize the pattern analyzer.
        
        Args:
            patterns_dir (str): Directory containing extracted patterns
            data_dir (str): Directory containing processed forex data
            output_dir (str, optional): Directory to save analysis results
        """
        self.patterns_dir = patterns_dir
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(os.path.dirname(patterns_dir), 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.patterns = {}
        self.processed_data = {}
        self.analysis_results = {}

    def load_patterns(self, timeframe):
        """
        Load extracted patterns for a specific timeframe.
        
        Args:
            timeframe (str): Timeframe to load patterns for
            
        Returns:
            dict: Loaded patterns data
        """
        patterns_file = os.path.join(self.patterns_dir, 'data', f"{timeframe}_patterns.json")
        full_patterns_file = os.path.join(self.patterns_dir, 'data', f"{timeframe}_full_patterns.pkl")
        
        if not os.path.exists(patterns_file) or not os.path.exists(full_patterns_file):
            logger.error(f"Pattern files not found for {timeframe} timeframe")
            return None
            
        logger.info(f"Loading patterns for {timeframe} timeframe")
        
        # Load pattern metadata
        with open(patterns_file, 'r') as f:
            patterns_meta = restore_json_infinity(json.load(f))

            
        # Load full pattern data
        with open(full_patterns_file, 'rb') as f:
            full_patterns = pickle.load(f)
            
        # Combine metadata and full data
        patterns_data = {
            'metadata': patterns_meta,
            'windows': full_patterns['windows'],
            'timestamps': full_patterns['timestamps'],
            'cluster_labels': full_patterns['cluster_labels']
        }
        
        self.patterns[timeframe] = patterns_data
        logger.info(f"Loaded {len(patterns_data['windows'])} patterns for {timeframe} timeframe")
        
        return patterns_data
    
    def load_processed_data(self, timeframe):
        """
        Load processed forex data for a specific timeframe.
        
        Args:
            timeframe (str): Timeframe to load data for
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        file_path = os.path.join(self.data_dir, f"XAU_{timeframe}_processed.csv")
        if not os.path.exists(file_path):
            logger.error(f"Processed data file not found: {file_path}")
            return None
            
        logger.info(f"Loading processed data for {timeframe} timeframe")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.processed_data[timeframe] = df
        logger.info(f"Loaded {len(df)} records for {timeframe} timeframe")
        
        return df
    
    def extract_pattern_features(self, timeframe):
        """
        Extract features from patterns for machine learning analysis.
        
        Args:
            timeframe (str): Timeframe to extract features for
            
        Returns:
            tuple: (X, y) where X is feature matrix and y is cluster labels
        """
        if timeframe not in self.patterns:
            logger.warning(f"No patterns loaded for {timeframe} timeframe")
            return None, None
            
        patterns_data = self.patterns[timeframe]
        windows = patterns_data['windows']
        cluster_labels = patterns_data['cluster_labels']
        
        # Extract features from each window
        features = []
        for window in tqdm(windows, desc=f"Extracting features from {timeframe} patterns"):
            # Basic statistical features
            open_prices = window[:, 0]
            high_prices = window[:, 1]
            low_prices = window[:, 2]
            close_prices = window[:, 3]
            
            # Calculate returns (avoid division by zero)
            if np.any(close_prices[:-1] == 0):
                # If any close price is zero, use simple diff instead
                returns = np.diff(close_prices)
            else:
                returns = np.diff(close_prices) / close_prices[:-1]
            
            # Calculate features
            window_features = [
                # Price movement
                (close_prices[-1] - close_prices[0]) / close_prices[0] if close_prices[0] != 0 else 0,  # Overall return
                np.mean(returns),  # Mean return
                np.std(returns),   # Volatility
                np.max(returns),   # Max return
                np.min(returns),   # Min return
                
                # Candlestick features
                np.mean(high_prices - low_prices),  # Average range
                np.mean(np.abs(close_prices - open_prices)),  # Average body size
                np.mean((high_prices - np.maximum(close_prices, open_prices))),  # Average upper shadow
                np.mean((np.minimum(close_prices, open_prices) - low_prices)),  # Average lower shadow
                
                # Trend features
                np.polyfit(np.arange(len(close_prices)), close_prices, 1)[0],  # Linear trend slope
                
                # Pattern shape features
                np.max(close_prices) - np.min(close_prices),  # Price range
                np.argmax(close_prices) / len(close_prices),  # Relative position of max
                np.argmin(close_prices) / len(close_prices),  # Relative position of min
                
                # Zigzag features
                np.sum(np.abs(np.diff(np.sign(np.diff(close_prices))))),  # Direction changes
            ]
            
            features.append(window_features)
            
        X = np.array(features)
        y = np.array(cluster_labels)
        
        # Handle NaN and infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning(f"Found {np.sum(np.isnan(X))} NaN and {np.sum(np.isinf(X))} infinite values in features, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} patterns")
        
        return X, y
    
    def reduce_dimensions(self, X, n_components=2):
        """
        Reduce dimensionality of feature matrix using PCA.
        
        Args:
            X (numpy.ndarray): Feature matrix
            n_components (int): Number of components to reduce to
            
        Returns:
            tuple: (X_reduced, pca) where X_reduced is reduced feature matrix and pca is fitted PCA model
        """
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        logger.info(f"Reduced dimensions from {X.shape[1]} to {n_components}")
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        return X_reduced, pca
    
    def cluster_analysis(self, X, y_original=None, n_clusters=None):
        """
        Perform cluster analysis on pattern features.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y_original (numpy.ndarray, optional): Original cluster labels
            n_clusters (int, optional): Number of clusters to form
            
        Returns:
            tuple: (y_pred, metrics) where y_pred is predicted cluster labels and metrics is evaluation metrics
        """
        if n_clusters is None and y_original is not None:
            n_clusters = len(np.unique(y_original))
        elif n_clusters is None:
            # Estimate optimal number of clusters
            n_clusters = max(2, int(np.sqrt(X.shape[0] / 5)))
            
        logger.info(f"Performing cluster analysis with {n_clusters} clusters")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)
        
        # Calculate metrics
        metrics = {
            'inertia': kmeans.inertia_
        }
        
        if X.shape[0] > n_clusters:
            metrics['silhouette'] = silhouette_score(X, y_pred)
            
        if y_original is not None:
            # Calculate cluster agreement metrics
            # Note: These are not traditional classification metrics, just measures of agreement
            contingency_table = pd.crosstab(y_original, y_pred)
            metrics['contingency_table'] = contingency_table.values.tolist()
            
        logger.info(f"Cluster analysis metrics: {metrics}")
        
        return y_pred, metrics
    
    def train_pattern_classifier(self, X, y, test_size=0.3):
        """
        Train a classifier to predict pattern clusters.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Cluster labels
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (model, metrics) where model is trained classifier and metrics is evaluation metrics
        """
        logger.info(f"Training pattern classifier with {X.shape[0]} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train random forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'feature_importance': model.feature_importances_.tolist()
        }
        
        logger.info(f"Classifier metrics: {metrics}")
        
        return model, metrics
    
    def detect_anomalous_patterns(self, X, contamination=0.05):
        """
        Detect anomalous patterns using Isolation Forest.
        
        Args:
            X (numpy.ndarray): Feature matrix
            contamination (float): Expected proportion of anomalies
            
        Returns:
            tuple: (anomalies, model) where anomalies is boolean array and model is fitted Isolation Forest
        """
        logger.info(f"Detecting anomalous patterns with contamination {contamination}")
        
        # Train isolation forest
        model = IsolationForest(contamination=contamination, random_state=42)
        y_pred = model.fit_predict(X)
        
        # Convert to boolean array (True for anomalies)
        anomalies = y_pred == -1
        
        logger.info(f"Detected {np.sum(anomalies)} anomalous patterns out of {len(anomalies)}")
        
        return anomalies, model
    
    def analyze_pattern_profitability(self, timeframe, lookahead_periods=10):
        """
        Analyze the profitability of identified patterns.
        
        Args:
            timeframe (str): Timeframe to analyze
            lookahead_periods (int): Number of periods to look ahead for returns
            
        Returns:
            dict: Profitability metrics by cluster
        """
        if timeframe not in self.patterns or timeframe not in self.processed_data:
            logger.warning(f"Missing data for {timeframe} timeframe")
            return None
            
        patterns_data = self.patterns[timeframe]
        df = self.processed_data[timeframe]
        
        windows = patterns_data['windows']
        timestamps = patterns_data['timestamps']
        cluster_labels = patterns_data['cluster_labels']
        
        logger.info(f"Analyzing profitability of {len(windows)} patterns with {lookahead_periods} period lookahead")
        
        # Calculate future returns for each pattern
        future_returns = []
        valid_indices = []
        
        for i, ts in tqdm(enumerate(timestamps), desc=f"Calculating future returns for {timeframe}"):
            # Convert timestamp string to datetime if needed
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
                
            # Find index of timestamp in dataframe
            try:
                idx = df.index.get_indexer([ts], method='nearest')[0]
                
                # Check if we have enough data for lookahead
                if idx + lookahead_periods < len(df):
                    # Calculate future return
                    future_return = (df.iloc[idx + lookahead_periods]['Close'] - df.iloc[idx]['Close']) / df.iloc[idx]['Close']
                    future_returns.append(future_return)
                    valid_indices.append(i)
                else:
                    logger.debug(f"Insufficient lookahead data for pattern at {ts}")
            except Exception as e:
                logger.warning(f"Error calculating future return for pattern at {ts}: {e}")
        
        if not future_returns:
            logger.error("No valid future returns calculated")
            return None
            
        # Convert to numpy arrays
        future_returns = np.array(future_returns)
        valid_cluster_labels = cluster_labels[valid_indices]
        
        # Calculate profitability metrics by cluster
        unique_clusters = np.unique(valid_cluster_labels)
        profitability = {}
        
        for cluster in unique_clusters:
            cluster_returns = future_returns[valid_cluster_labels == cluster]
            
            if len(cluster_returns) < 5:
                logger.warning(f"Insufficient data for cluster {cluster}")
                continue
                
            # Calculate metrics
            mean_return = np.mean(cluster_returns)
            median_return = np.median(cluster_returns)
            std_return = np.std(cluster_returns)
            positive_rate = np.mean(cluster_returns > 0)
            
            # Calculate profit factor for the cluster
            sum_positive = np.sum(cluster_returns[cluster_returns > 0])
            sum_negative = np.sum(np.abs(cluster_returns[cluster_returns < 0]))
            profit_factor = (sum_positive / sum_negative) if sum_negative > 0 else float('inf')
            
            # Statistical significance (t-test against 0)
            t_stat, p_value = ttest_1samp(cluster_returns, 0)
            
            profitability[int(cluster)] = {
                'count': len(cluster_returns),
                'mean_return': float(mean_return),
                'avg_return': float(mean_return),
                'median_return': float(median_return),
                'std_return': float(std_return),
                'win_rate': float(positive_rate),
                'profit_factor': float(profit_factor),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
            
        logger.info(f"Calculated profitability metrics for {len(profitability)} clusters")
        
        return profitability
    
    def compare_pattern_profitability(self, timeframe, profitability):
        """
        Compare profitability between different pattern clusters.
        
        Args:
            timeframe (str): Timeframe to analyze
            profitability (dict): Profitability metrics by cluster
            
        Returns:
            dict: Comparison results
        """
        if not profitability:
            logger.warning("No profitability data to compare")
            return None
            
        logger.info(f"Comparing profitability between {len(profitability)} clusters")
        
        # Find most and least profitable clusters
        clusters = list(profitability.keys())
        mean_returns = [profitability[c]['mean_return'] for c in clusters]
        
        most_profitable_idx = np.argmax(mean_returns)
        least_profitable_idx = np.argmin(mean_returns)
        
        most_profitable = clusters[most_profitable_idx]
        least_profitable = clusters[least_profitable_idx]
        
        # Statistical comparison between most and least profitable
        comparison = {
            'most_profitable_cluster': int(most_profitable),
            'most_profitable_return': profitability[most_profitable]['mean_return'],
            'least_profitable_cluster': int(least_profitable),
            'least_profitable_return': profitability[least_profitable]['mean_return'],
            'return_difference': profitability[most_profitable]['mean_return'] - profitability[least_profitable]['mean_return']
        }
        
        # Significant clusters (p < 0.05)
        significant_clusters = [c for c in clusters if profitability[c]['significant']]
        significant_positive = [c for c in significant_clusters if profitability[c]['mean_return'] > 0]
        significant_negative = [c for c in significant_clusters if profitability[c]['mean_return'] < 0]
        
        comparison['significant_clusters_count'] = len(significant_clusters)
        comparison['significant_positive_count'] = len(significant_positive)
        comparison['significant_negative_count'] = len(significant_negative)
        
        if significant_positive:
            best_sig_idx = np.argmax([profitability[c]['mean_return'] for c in significant_positive])
            best_sig_cluster = significant_positive[best_sig_idx]
            comparison['best_significant_cluster'] = int(best_sig_cluster)
            comparison['best_significant_return'] = profitability[best_sig_cluster]['mean_return']
            comparison['best_significant_p_value'] = profitability[best_sig_cluster]['p_value']
        
        logger.info(f"Comparison results: {comparison}")
        
        return comparison
    
    def visualize_cluster_profitability(self, timeframe, profitability):
        """
        Visualize profitability of different pattern clusters.
        
        Args:
            timeframe (str): Timeframe being analyzed
            profitability (dict): Profitability metrics by cluster
            
        Returns:
            str: Path to saved visualization
        """
        if not profitability:
            logger.warning("No profitability data to visualize")
            return None
            
        visualization_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Create bar chart of mean returns by cluster
        clusters = list(profitability.keys())
        mean_returns = [profitability[c]['mean_return'] for c in clusters]
        p_values = [profitability[c]['p_value'] for c in clusters]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by mean return
        sorted_indices = np.argsort(mean_returns)
        sorted_clusters = [clusters[i] for i in sorted_indices]
        sorted_returns = [mean_returns[i] for i in sorted_indices]
        sorted_p_values = [p_values[i] for i in sorted_indices]
        
        # Create bar colors based on statistical significance
        colors = ['green' if profitability[c]['significant'] and profitability[c]['mean_return'] > 0 else
                 'red' if profitability[c]['significant'] and profitability[c]['mean_return'] < 0 else
                 'lightgreen' if profitability[c]['mean_return'] > 0 else 'lightcoral'
                 for c in sorted_clusters]
        
        bars = ax.bar(range(len(sorted_clusters)), sorted_returns, color=colors)
        
        # Add cluster labels
        ax.set_xticks(range(len(sorted_clusters)))
        ax.set_xticklabels([f"C{c}" for c in sorted_clusters], rotation=90)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels and title
        ax.set_xlabel('Pattern Cluster')
        ax.set_ylabel('Mean Return')
        ax.set_title(f'Pattern Profitability by Cluster ({timeframe} timeframe)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Significant Positive'),
            Patch(facecolor='lightgreen', label='Non-significant Positive'),
            Patch(facecolor='red', label='Significant Negative'),
            Patch(facecolor='lightcoral', label='Non-significant Negative')
        ]
        ax.legend(handles=legend_elements)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save figure
        file_path = os.path.join(visualization_dir, f"{timeframe}_cluster_profitability.png")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close(fig)
        
        logger.info(f"Saved profitability visualization to {file_path}")
        
        return file_path
    
    def visualize_feature_importance(self, feature_importance, feature_names=None):
        """
        Visualize feature importance from classifier.
        
        Args:
            feature_importance (array-like): Feature importance scores
            feature_names (list, optional): Names of features
            
        Returns:
            str: Path to saved visualization
        """
        visualization_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(feature_importance))]
            
        # Sort by importance
        sorted_indices = np.argsort(feature_importance)
        sorted_importance = [feature_importance[i] for i in sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_importance, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for Pattern Classification')
        
        # Save figure
        file_path = os.path.join(visualization_dir, "feature_importance.png")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close(fig)
        
        logger.info(f"Saved feature importance visualization to {file_path}")
        
        return file_path
    
    def visualize_pca_clusters(self, X_reduced, y, title=None):
        """
        Visualize clusters in reduced PCA space.
        
        Args:
            X_reduced (numpy.ndarray): Reduced feature matrix (2D)
            y (numpy.ndarray): Cluster labels
            title (str, optional): Plot title
            
        Returns:
            str: Path to saved visualization
        """
        visualization_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique clusters
        unique_clusters = np.unique(y)
        
        # Create colormap
        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        # Plot each cluster
        for i, cluster in enumerate(unique_clusters):
            mask = y == cluster
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                      label=f'Cluster {cluster}', color=colors[i], alpha=0.7)
            
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Pattern Clusters in PCA Space')
            
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        file_path = os.path.join(visualization_dir, "pca_clusters.png")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close(fig)
        
        logger.info(f"Saved PCA clusters visualization to {file_path}")
        
        return file_path
    
    def save_analysis_results(self, timeframe, results):
        """
        Save analysis results to disk.
        
        Args:
            timeframe (str): Timeframe of the analysis
            results (dict): Analysis results
            
        Returns:
            str: Path to saved results file
        """
        results_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(results_dir, exist_ok=True)

        # Convert all NumPy types to native Python types
        #print(results)
        results_serializable = make_json_serializable(results)

        # Add metadata
        results_serializable['timeframe'] = timeframe
        results_serializable['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save results
        results_file = os.path.join(results_dir, f"{timeframe}_analysis.json")
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
            
        logger.info(f"Saved analysis results to {results_file}")
        
        return results_file
    
    def analyze_patterns(self, timeframe, lookahead_periods=10):
        """
        Perform comprehensive analysis of patterns for a specific timeframe.
        
        Args:
            timeframe (str): Timeframe to analyze
            lookahead_periods (int): Number of periods to look ahead for returns
            
        Returns:
            dict: Analysis results
        """
        # Load data if not already loaded
        if timeframe not in self.patterns:
            self.load_patterns(timeframe)
            
        if timeframe not in self.processed_data:
            self.load_processed_data(timeframe)
            
        if timeframe not in self.patterns or timeframe not in self.processed_data:
            logger.error(f"Failed to load data for {timeframe} timeframe")
            return None
            
        logger.info(f"Analyzing patterns for {timeframe} timeframe")
        
        # Extract features
        X, y = self.extract_pattern_features(timeframe)
        
        if X is None or y is None:
            logger.error("Feature extraction failed")
            return None
            
        # Reduce dimensions for visualization
        X_reduced, pca = self.reduce_dimensions(X, n_components=2)
        
        # Cluster analysis
        y_pred, cluster_metrics = self.cluster_analysis(X, y)
        
        # Train classifier
        model, classifier_metrics = self.train_pattern_classifier(X, y)
        
        # Detect anomalies
        anomalies, anomaly_model = self.detect_anomalous_patterns(X)
        
        # Analyze profitability
        profitability = self.analyze_pattern_profitability(timeframe, lookahead_periods)
        
        # Compute top-level profitability summary and add cluster_returns
        if profitability:
            # Add avg_return alias to each cluster (already done in analyze_pattern_profitability)
            
            # Calculate weighted avg_return
            total_count = sum([profitability[c]['count'] for c in profitability])
            if total_count > 0:
                weighted_avg_return = sum([profitability[c]['avg_return'] * profitability[c]['count'] for c in profitability]) / total_count
                weighted_win_rate = sum([profitability[c]['win_rate'] * profitability[c]['count'] for c in profitability]) / total_count
            else:
                weighted_avg_return = 0.0
                weighted_win_rate = 0.0
            
            profitable_patterns = sum(1 for c in profitability if profitability[c]['avg_return'] > 0)
            
            # Calculate profit_factor as ratio of sum positive returns to sum negative returns
            sum_positive = 0.0
            sum_negative = 0.0
            for c in profitability:
                cluster_returns = [profitability[c]['avg_return']] * profitability[c]['count']  # approximate
                for r in cluster_returns:
                    if r > 0:
                        sum_positive += r
                    else:
                        sum_negative += abs(r)
            profit_factor = (sum_positive / sum_negative) if sum_negative > 0 else float('inf')
            
            top_level_profitability = {
                'avg_return': weighted_avg_return,
                'profitable_patterns': profitable_patterns,
                'win_rate': weighted_win_rate,
                'profit_factor': profit_factor
            }
            
            # Compare profitability
            comparison = self.compare_pattern_profitability(timeframe, profitability)
            
            # Visualize profitability
            profitability_viz = self.visualize_cluster_profitability(timeframe, profitability)
        else:
            top_level_profitability = None
            comparison = None
            profitability_viz = None
            
        # Visualize feature importance
        feature_names = [
            'Overall return', 'Mean return', 'Volatility', 'Max return', 'Min return',
            'Average range', 'Average body size', 'Average upper shadow', 'Average lower shadow',
            'Linear trend slope', 'Price range', 'Relative max position', 'Relative min position',
            'Direction changes'
        ]
        importance_viz = self.visualize_feature_importance(classifier_metrics['feature_importance'], feature_names)
        
        # Visualize PCA clusters
        pca_viz = self.visualize_pca_clusters(X_reduced, y, f"Pattern Clusters for {timeframe} Timeframe")
        
        # Compile results
        results = {
            'feature_extraction': {
                'n_patterns': X.shape[0],
                'n_features': X.shape[1]
            },
            'pca': {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'visualization': pca_viz
            },
            'clustering': cluster_metrics,
            'classification': {
                'accuracy': classifier_metrics['accuracy'],
                'precision': classifier_metrics['precision'],
                'recall': classifier_metrics['recall'],
                'f1': classifier_metrics['f1'],
                'feature_importance': classifier_metrics['feature_importance'],
                'feature_importance_viz': importance_viz
            },
            'anomaly_detection': {
                'n_anomalies': int(np.sum(anomalies)),
                'anomaly_rate': float(np.mean(anomalies))
            },
            'profitability': top_level_profitability,
            'cluster_returns': profitability,
            'profitability_comparison': comparison,
            'profitability_viz': profitability_viz,
            'statistical_significance': {}
        }

        # Calculate statistical significance data for each cluster
        if profitability:
            for cluster_id, data in profitability.items():
                results['statistical_significance'][cluster_id] = {
                    'p_value': data.get('p_value', 1.0),
                    'significant': data.get('significant', False)
                }
        
        # Save results
        results_file = self.save_analysis_results(timeframe, results)
        results['results_file'] = results_file
        
        # Store results
        self.analysis_results[timeframe] = results
        
        logger.info(f"Analysis complete for {timeframe} timeframe")
        
        return results
    
    def analyze_all_timeframes(self, timeframes=None, **kwargs):
        """
        Analyze patterns for all available timeframes.
        
        Args:
            timeframes (list, optional): List of timeframes to analyze
            **kwargs: Additional arguments for analyze_patterns
            
        Returns:
            dict: Dictionary of results by timeframe
        """
        if timeframes is None:
            # Find all pattern files
            pattern_dir = os.path.join(self.patterns_dir, 'data')
            if os.path.exists(pattern_dir):
                files = os.listdir(pattern_dir)
                timeframes = [f.replace('_patterns.json', '') 
                             for f in files if f.endswith('_patterns.json')]
            else:
                logger.error(f"Pattern directory not found: {pattern_dir}")
                return None
            
        logger.info(f"Analyzing patterns for {len(timeframes)} timeframes: {timeframes}")
        
        results = {}
        for tf in timeframes:
            logger.info(f"Processing {tf} timeframe")
            tf_results = self.analyze_patterns(tf, **kwargs)
            results[tf] = tf_results
            
        return results


def main():
    """
    Main function to demonstrate the pattern analysis workflow.
    """
    # Set directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    patterns_dir = os.path.join(base_dir, 'data', 'patterns')
    output_dir = os.path.join(base_dir, 'data', 'analysis')
    
    # Initialize pattern analyzer
    analyzer = PatternAnalyzer(patterns_dir, data_dir, output_dir)
    
    # Analyze patterns for 1h timeframe
    timeframe = '1h'
    lookahead_periods = 12  # 12 periods (hours) lookahead for returns
    
    analyzer.analyze_patterns(timeframe, lookahead_periods=lookahead_periods)
    
    logger.info("Pattern analysis completed successfully")

if __name__ == "__main__":
    main()
