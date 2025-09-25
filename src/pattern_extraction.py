#!/usr/bin/env python3
"""
Pattern Extraction Module for Forex Pattern Discovery Framework
This module implements methods for extracting and representing candlestick patterns.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pattern_extraction')

class TemplateGrid:
    """
    Implementation of the Template Grid (TG) approach for capturing chart formations.
    """
    
    def __init__(self, rows=10, cols=10):
        """
        Initialize a Template Grid with specified dimensions.
        
        Args:
            rows (int): Number of rows in the grid (price dimension)
            cols (int): Number of columns in the grid (time dimension)
        """
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols))
        
    def fit(self, prices):
        """
        Fit the Template Grid to a price series.
        
        Args:
            prices (array-like): Array of price values to fit
            
        Returns:
            numpy.ndarray: Filled Template Grid
        """
        if len(prices) < self.cols:
            logger.warning(f"Price series length ({len(prices)}) is less than grid columns ({self.cols})")
            # Pad with last value if needed
            prices = np.pad(prices, (0, self.cols - len(prices)), 'edge')
        elif len(prices) > self.cols:
            # Resample to fit grid columns
            indices = np.linspace(0, len(prices) - 1, self.cols)
            prices = np.array([prices[int(i)] for i in indices])
        
        # Normalize to fit grid rows
        min_price = np.min(prices)
        max_price = np.max(prices)
        if max_price == min_price:
            # Handle flat price series
            normalized = np.ones(len(prices)) * (self.rows // 2)
        else:
            normalized = ((prices - min_price) / (max_price - min_price)) * (self.rows - 1)
        
        # Fill grid
        self.grid = np.zeros((self.rows, self.cols))
        for col, price_idx in enumerate(normalized):
            row = int(self.rows - 1 - price_idx)  # Invert row index (0 is top)
            row = max(0, min(row, self.rows - 1))  # Ensure within bounds
            self.grid[row, col] = 1
            
        return self.grid
    
    def generate_pic(self):
        """
        Generate a Pattern Identification Code (PIC) from the grid.
        
        Returns:
            list: One-dimensional array representing the pattern
        """
        # Flatten grid to 1D array
        return self.grid.flatten().tolist()
    
    def visualize(self, title=None, save_path=None):
        """
        Visualize the Template Grid.
        
        Args:
            title (str, optional): Title for the plot
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(self.grid, cmap='Blues', interpolation='nearest')

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

        return fig


class PatternExtractor:
    """
    Class for extracting patterns from forex data using various methods.
    """
    
    def __init__(self, data_dir, output_dir=None):
        """
        Initialize the pattern extractor.
        
        Args:
            data_dir (str): Directory containing processed forex data
            output_dir (str, optional): Directory to save extracted patterns
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(os.path.dirname(data_dir), 'patterns')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.processed_data = {}
        self.patterns = {}
        self.pattern_clusters = {}
        
    def load_data(self, timeframe):
        """
        Load processed data for a specific timeframe.
        
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
    
    def extract_candlestick_windows(self, timeframe, window_size=5, stride=1, max_windows=None):
        """
        Extract sliding windows of candlestick data.
        
        Args:
            timeframe (str): Timeframe to extract windows from
            window_size (int): Size of each window
            stride (int): Step size between windows
            max_windows (int, optional): Maximum number of windows to extract
            
        Returns:
            tuple: (windows, timestamps) where windows is a list of arrays and timestamps are end times
        """
        if timeframe not in self.processed_data:
            logger.warning(f"No data loaded for {timeframe} timeframe")
            return None, None
            
        df = self.processed_data[timeframe]
        
        # Select OHLC columns for pattern extraction
        ohlc_data = df[['Open', 'High', 'Low', 'Close']].values
        
        windows = []
        timestamps = []
        
        for i in tqdm(range(0, len(df) - window_size + 1, stride), desc=f"Extracting windows from {timeframe}"):
            if max_windows and len(windows) >= max_windows:
                break
                
            window = ohlc_data[i:i+window_size]
            windows.append(window)
            timestamps.append(df.index[i+window_size-1])
            
        logger.info(f"Extracted {len(windows)} windows of size {window_size} from {timeframe} data")
        
        return windows, timestamps
    
    def create_template_grids(self, windows, timestamps, grid_rows=10, grid_cols=10):
        """
        Create Template Grids for a set of windows.
        
        Args:
            windows (list): List of price windows
            timestamps (list): List of corresponding timestamps
            grid_rows (int): Number of rows in the Template Grid
            grid_cols (int): Number of columns in the Template Grid
            
        Returns:
            tuple: (grids, pics) where grids is a list of Template Grid objects and pics is a list of PICs
        """
        grids = []
        pics = []
        
        for window in tqdm(windows, desc="Creating Template Grids"):
            # Extract close prices for the grid
            close_prices = window[:, 3]  # Close is at index 3 in OHLC
            
            # Create and fit grid
            grid = TemplateGrid(rows=grid_rows, cols=grid_cols)
            grid.fit(close_prices)
            
            # Generate PIC
            pic = grid.generate_pic()
            
            grids.append(grid)
            pics.append(pic)
            
        logger.info(f"Created {len(grids)} Template Grids with dimensions {grid_rows}x{grid_cols}")
        
        return grids, pics
    
    def calculate_dtw_distance_matrix(self, windows):
        """
        Calculate distance matrix using Dynamic Time Warping.
        
        Args:
            windows (list): List of price windows
            
        Returns:
            numpy.ndarray: Distance matrix
        """
        n_windows = len(windows)
        distance_matrix = np.zeros((n_windows, n_windows))
        
        logger.info(f"Calculating DTW distance matrix for {n_windows} windows")
        
        # Extract close prices for DTW
        close_prices = [window[:, 3] for window in windows]
        
        # Calculate DTW distances
        for i in tqdm(range(n_windows), desc="Calculating DTW distances"):
            for j in range(i+1, n_windows):
                distance = dtw(close_prices[i], close_prices[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                
        return distance_matrix
    
    def calculate_pic_similarity_matrix(self, pics):
        """
        Calculate similarity matrix based on Pattern Identification Codes.
        
        Args:
            pics (list): List of Pattern Identification Codes
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        n_pics = len(pics)
        similarity_matrix = np.zeros((n_pics, n_pics))
        
        logger.info(f"Calculating PIC similarity matrix for {n_pics} patterns")
        
        for i in tqdm(range(n_pics), desc="Calculating PIC similarities"):
            for j in range(i, n_pics):
                # Calculate similarity as 1 - normalized Euclidean distance
                distance = np.sqrt(np.sum((np.array(pics[i]) - np.array(pics[j]))**2))
                max_possible_distance = np.sqrt(len(pics[i]))  # Maximum possible distance
                similarity = 1 - (distance / max_possible_distance)
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                
        return similarity_matrix
    
    def cluster_patterns(self, distance_matrix, n_clusters=None, distance_threshold=None):
        """
        Cluster patterns using hierarchical clustering.
        
        Args:
            distance_matrix (numpy.ndarray): Distance matrix between patterns
            n_clusters (int, optional): Number of clusters to form
            distance_threshold (float, optional): Distance threshold for clustering
            
        Returns:
            numpy.ndarray: Cluster labels for each pattern
        """
        if n_clusters is None and distance_threshold is None:
            # Estimate optimal number of clusters
            n_clusters = max(2, int(np.sqrt(distance_matrix.shape[0] / 2)))
            
        logger.info(f"Clustering patterns into {n_clusters} clusters" if n_clusters else 
                   f"Clustering patterns with distance threshold {distance_threshold}")
        
        # Create clustering model
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        # Fit model
        cluster_labels = model.fit_predict(distance_matrix)
        
        # Count patterns per cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            logger.info(f"Cluster {label}: {count} patterns")
            
        return cluster_labels
    
    def extract_representative_patterns(self, windows, timestamps, cluster_labels, distance_matrix):
        """
        Extract representative patterns from each cluster.
        
        Args:
            windows (list): List of price windows
            timestamps (list): List of corresponding timestamps
            cluster_labels (numpy.ndarray): Cluster labels for each pattern
            distance_matrix (numpy.ndarray): Distance matrix between patterns
            
        Returns:
            dict: Dictionary of representative patterns by cluster
        """
        unique_labels = np.unique(cluster_labels)
        representatives = {}
        
        for label in unique_labels:
            # Get indices of patterns in this cluster
            cluster_indices = np.where(cluster_labels == label)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Find the pattern with minimum average distance to others in the cluster
            min_avg_distance = float('inf')
            representative_idx = None
            
            for idx in cluster_indices:
                # Calculate average distance to other patterns in the cluster
                distances = [distance_matrix[idx, other_idx] for other_idx in cluster_indices if other_idx != idx]
                avg_distance = np.mean(distances) if distances else 0
                
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    representative_idx = idx
            
            # Store representative pattern
            representatives[str(label)] = {
                'window': windows[representative_idx],
                'timestamp': timestamps[representative_idx],
                'index': int(representative_idx),
                'count': int(len(cluster_indices))
            }
            
        logger.info(f"Extracted {len(representatives)} representative patterns")
        
        return representatives
    
    def visualize_representative_patterns(self, timeframe, representatives, grid_rows=10, grid_cols=10):
        """
        Visualize representative patterns.
        
        Args:
            timeframe (str): Timeframe of the patterns
            representatives (dict): Dictionary of representative patterns by cluster
            grid_rows (int): Number of rows in the Template Grid
            grid_cols (int): Number of columns in the Template Grid
            
        Returns:
            dict: Dictionary of visualization file paths by cluster
        """
        visualization_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        
        visualization_files = {}
        
        for label, rep in representatives.items():
            window = rep['window']
            timestamp = rep['timestamp']
            count = rep['count']
            
            # Create Template Grid visualization
            grid = TemplateGrid(rows=grid_rows, cols=grid_cols)
            grid.fit(window[:, 3])  # Close prices
            
            # Save visualization
            file_path = os.path.join(visualization_dir, f"{timeframe}_pattern_{label}.png")
            title = f"Cluster {label} Representative Pattern\n({count} occurrences)"
            grid.visualize(title=title, save_path=file_path)
            
            # Create candlestick visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate x-axis values
            x = np.arange(len(window))
            
            # Plot candlesticks
            for i, (open_price, high, low, close) in enumerate(window):
                # Determine if bullish or bearish
                color = 'green' if close >= open_price else 'red'
                
                # Plot the body
                ax.plot([i, i], [open_price, close], color=color, linewidth=2)
                
                # Plot the wicks
                ax.plot([i, i], [low, high], color='black', linewidth=1)
            
            ax.set_title(f"Cluster {label} Representative Candlestick Pattern\n({count} occurrences)")
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            
            # Save candlestick visualization
            candle_file_path = os.path.join(visualization_dir, f"{timeframe}_candlestick_{label}.png")
            plt.savefig(candle_file_path, bbox_inches='tight')
            plt.close(fig)
            
            visualization_files[label] = {
                'grid': file_path,
                'candlestick': candle_file_path
            }
            
        logger.info(f"Saved visualizations for {len(visualization_files)} representative patterns")
        
        return visualization_files
    
    def save_patterns(self, timeframe, windows, timestamps, cluster_labels, representatives):
        """
        Save extracted patterns to disk.
        
        Args:
            timeframe (str): Timeframe of the patterns
            windows (list): List of price windows
            timestamps (list): List of corresponding timestamps
            cluster_labels (numpy.ndarray): Cluster labels for each pattern
            representatives (dict): Dictionary of representative patterns by cluster
            
        Returns:
            str: Path to saved patterns file
        """
        patterns_dir = os.path.join(current_app.config['PATTERNS_FOLDER'], 'data')
        os.makedirs(patterns_dir, exist_ok=True)

        # Convert timestamps to strings for JSON serialization
        timestamps_str = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]

        # Create serializable representatives dictionary
        serializable_representatives = {}
        for label, rep_data in representatives.items():
            serializable_representatives[label] = {
                'window': rep_data['window'].tolist() if isinstance(rep_data['window'], np.ndarray) else rep_data['window'],
                'timestamp': rep_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(rep_data['timestamp'], 'strftime') else rep_data['timestamp'],
                'index': int(rep_data['index']),
                'count': int(rep_data['count'])
            }

        # Create patterns dictionary
        patterns_data = {
            'timeframe': timeframe,
            'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_patterns': len(windows),
            'window_size': len(windows[0]) if windows else 0,
            'cluster_labels': cluster_labels.tolist(),
            'representatives': serializable_representatives,
            'unique_clusters': int(len(np.unique(cluster_labels)))
        }

        # Save patterns data
        patterns_file = os.path.join(patterns_dir, f"{timeframe}_patterns.json")
        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)

        # Save full pattern data (windows and timestamps) as pickle
        full_data = {
            'windows': windows,
            'timestamps': timestamps,
            'cluster_labels': cluster_labels
        }

        pickle_file = os.path.join(patterns_dir, f"{timeframe}_full_patterns.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(full_data, f)
            
        logger.info(f"Saved patterns data to {patterns_file} and {pickle_file}")
        
        return patterns_file
    
    def extract_patterns(self, timeframe, window_size=5, stride=1, max_windows=10000, 
                        grid_rows=10, grid_cols=10, n_clusters=None):
        """
        Extract patterns from a specific timeframe.
        
        Args:
            timeframe (str): Timeframe to extract patterns from
            window_size (int): Size of each pattern window
            stride (int): Step size between windows
            max_windows (int): Maximum number of windows to extract
            grid_rows (int): Number of rows in the Template Grid
            grid_cols (int): Number of columns in the Template Grid
            n_clusters (int, optional): Number of clusters to form
            
        Returns:
            tuple: (patterns_file, visualization_files) paths to saved files
        """
        # Load data if not already loaded
        if timeframe not in self.processed_data:
            self.load_data(timeframe)
            
        if timeframe not in self.processed_data:
            logger.error(f"Failed to load data for {timeframe} timeframe")
            return None, None
            
        logger.info(f"Extracting patterns from {timeframe} timeframe")
        
        # Extract windows
        windows, timestamps = self.extract_candlestick_windows(
            timeframe, window_size=window_size, stride=stride, max_windows=max_windows
        )
        
        if not windows:
            logger.error(f"No windows extracted from {timeframe} timeframe")
            return None, None
            
        # Create Template Grids and PICs
        grids, pics = self.create_template_grids(windows, timestamps, grid_rows=grid_rows, grid_cols=grid_cols)
        
        # Calculate distance matrix using DTW
        distance_matrix = self.calculate_dtw_distance_matrix(windows)
        
        # Calculate similarity matrix based on PICs
        similarity_matrix = self.calculate_pic_similarity_matrix(pics)
        
        # Convert similarity to distance
        pic_distance_matrix = 1 - similarity_matrix
        
        # Combine distance matrices (equal weight)
        combined_distance_matrix = (distance_matrix + pic_distance_matrix) / 2
        
        # Cluster patterns
        if n_clusters is None:
            # Estimate optimal number of clusters
            n_clusters = max(5, int(np.sqrt(len(windows) / 5)))
            
        cluster_labels = self.cluster_patterns(combined_distance_matrix, n_clusters=n_clusters)
        
        # Extract representative patterns
        representatives = self.extract_representative_patterns(
            windows, timestamps, cluster_labels, combined_distance_matrix
        )
        
        # Visualize representative patterns
        visualization_files = self.visualize_representative_patterns(
            timeframe, representatives, grid_rows=grid_rows, grid_cols=grid_cols
        )
        
        # Save patterns
        patterns_file = self.save_patterns(timeframe, windows, timestamps, cluster_labels, representatives)
        
        # Store patterns data
        self.patterns[timeframe] = {
            'windows': windows,
            'timestamps': timestamps,
            'cluster_labels': cluster_labels,
            'representatives': representatives
        }
        
        logger.info(f"Pattern extraction complete for {timeframe} timeframe")
        
        return patterns_file, visualization_files
    
    def extract_patterns_all_timeframes(self, timeframes=None, **kwargs):
        """
        Extract patterns from all available timeframes.
        
        Args:
            timeframes (list, optional): List of timeframes to process
            **kwargs: Additional arguments for extract_patterns
            
        Returns:
            dict: Dictionary of results by timeframe
        """
        if timeframes is None:
            # Find all processed data files
            files = os.listdir(self.data_dir)
            timeframes = [f.replace('XAU_', '').replace('_processed.csv', '') 
                         for f in files if f.startswith('XAU_') and f.endswith('_processed.csv')]
            
        logger.info(f"Extracting patterns from {len(timeframes)} timeframes: {timeframes}")
        
        results = {}
        for tf in timeframes:
            logger.info(f"Processing {tf} timeframe")
            patterns_file, visualization_files = self.extract_patterns(tf, **kwargs)
            results[tf] = {
                'patterns_file': patterns_file,
                'visualization_files': visualization_files
            }
            
        return results


class PiecewiseLinearRegression:
    """
    Implementation of Piecewise Linear Regression for trend identification.
    """
    
    def __init__(self, max_segments=5, min_segment_size=5):
        """
        Initialize the PLR model.
        
        Args:
            max_segments (int): Maximum number of segments to fit
            min_segment_size (int): Minimum number of points in each segment
        """
        self.max_segments = max_segments
        self.min_segment_size = min_segment_size
        self.segments = []
        
    def fit(self, x, y):
        """
        Fit piecewise linear regression to data.
        
        Args:
            x (array-like): X values (typically time indices)
            y (array-like): Y values (typically prices)
            
        Returns:
            list: List of segment information (start, end, slope, intercept)
        """
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        
        if n < 2 * self.min_segment_size:
            # Not enough data for multiple segments
            slope, intercept = np.polyfit(x, y, 1)
            self.segments = [{
                'start': 0,
                'end': n - 1,
                'slope': slope,
                'intercept': intercept
            }]
            return self.segments
            
        # Dynamic programming approach to find optimal segmentation
        # Cost matrix: cost[i][j] is the cost of fitting a line from i to j
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(i + self.min_segment_size - 1, n):
                segment_x = x[i:j+1]
                segment_y = y[i:j+1]
                slope, intercept = np.polyfit(segment_x, segment_y, 1)
                predicted = slope * segment_x + intercept
                cost[i, j] = np.sum((segment_y - predicted) ** 2)
        
        # Find optimal segmentation
        dp = np.full(n, float('inf'))  # dp[i] is the minimum cost up to point i
        prev = np.zeros(n, dtype=int)  # prev[i] is the end of the previous segment
        
        dp[self.min_segment_size - 1] = cost[0, self.min_segment_size - 1]
        
        for i in range(self.min_segment_size, n):
            for j in range(self.min_segment_size - 1, i):
                if dp[j] + cost[j + 1, i] < dp[i]:
                    dp[i] = dp[j] + cost[j + 1, i]
                    prev[i] = j
        
        # Reconstruct segments
        segments = []
        end = n - 1
        
        while end >= 0:
            start = prev[end] + 1 if end > 0 else 0
            segment_x = x[start:end+1]
            segment_y = y[start:end+1]
            
            if len(segment_x) >= self.min_segment_size:
                slope, intercept = np.polyfit(segment_x, segment_y, 1)
                segments.append({
                    'start': start,
                    'end': end,
                    'slope': slope,
                    'intercept': intercept
                })
                
            end = prev[end]
            
            if len(segments) >= self.max_segments:
                break
        
        self.segments = list(reversed(segments))
        return self.segments
    
    def predict(self, x):
        """
        Predict values using the fitted piecewise linear model.
        
        Args:
            x (array-like): X values to predict for
            
        Returns:
            numpy.ndarray: Predicted Y values
        """
        x = np.array(x)
        y_pred = np.zeros_like(x, dtype=float)
        
        for i, xi in enumerate(x):
            # Find the appropriate segment
            for segment in self.segments:
                start_idx = segment['start']
                end_idx = segment['end']
                
                if start_idx <= i <= end_idx:
                    y_pred[i] = segment['slope'] * xi + segment['intercept']
                    break
        
        return y_pred
    
    def visualize(self, x, y, title=None, save_path=None):
        """
        Visualize the piecewise linear regression.
        
        Args:
            x (array-like): X values
            y (array-like): Y values
            title (str, optional): Title for the plot
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        x = np.array(x)
        y = np.array(y)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original data
        ax.scatter(x, y, s=10, alpha=0.5, label='Original data')

        # Plot segments
        for segment in self.segments:
            start_idx = segment['start']
            end_idx = segment['end']
            segment_x = x[start_idx:end_idx+1]
            predicted = segment['slope'] * segment_x + segment['intercept']

            ax.plot(segment_x, predicted, linewidth=2,
                   label=f"Segment {start_idx}-{end_idx} (slope: {segment['slope']:.4f})")

        if title:
            ax.set_title(title)

        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

        return fig


class GeneticAlgorithm:
    """
    Implementation of Genetic Algorithm for pattern optimization.
    """
    
    def __init__(self, population_size=100, generations=50, mutation_rate=0.1, crossover_rate=0.7):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            population_size (int): Size of the population
            generations (int): Number of generations to evolve
            mutation_rate (float): Probability of mutation
            crossover_rate (float): Probability of crossover
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def initialize_population(self, grid_rows, grid_cols, n_patterns=None):
        """
        Initialize a population of random patterns.
        
        Args:
            grid_rows (int): Number of rows in the Template Grid
            grid_cols (int): Number of columns in the Template Grid
            n_patterns (int, optional): Number of patterns to initialize
            
        Returns:
            list: List of random patterns
        """
        n_patterns = n_patterns or self.population_size
        population = []
        
        for _ in range(n_patterns):
            # Create random grid
            grid = np.zeros((grid_rows, grid_cols))
            
            # Add random path through the grid
            row = np.random.randint(0, grid_rows)
            for col in range(grid_cols):
                grid[row, col] = 1
                # Randomly move up, stay, or move down
                move = np.random.choice([-1, 0, 1])
                row = max(0, min(grid_rows - 1, row + move))
            
            population.append(grid.flatten().tolist())
            
        return population
    
    def fitness(self, pattern, reference_patterns, similarity_threshold=0.7):
        """
        Calculate fitness of a pattern based on similarity to reference patterns.
        
        Args:
            pattern (list): Pattern to evaluate
            reference_patterns (list): List of reference patterns
            similarity_threshold (float): Threshold for considering patterns similar
            
        Returns:
            float: Fitness score
        """
        if not reference_patterns:
            return 0
            
        similarities = []
        
        for ref_pattern in reference_patterns:
            # Calculate similarity as 1 - normalized Euclidean distance
            distance = np.sqrt(np.sum((np.array(pattern) - np.array(ref_pattern))**2))
            max_possible_distance = np.sqrt(len(pattern))  # Maximum possible distance
            similarity = 1 - (distance / max_possible_distance)
            similarities.append(similarity)
        
        # Count patterns above similarity threshold
        similar_count = sum(1 for s in similarities if s >= similarity_threshold)
        
        # Fitness is the proportion of similar patterns
        fitness = similar_count / len(reference_patterns)
        
        return fitness
    
    def select_parents(self, population, fitnesses):
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            population (list): List of patterns
            fitnesses (list): List of fitness scores
            
        Returns:
            tuple: (parent1, parent2)
        """
        # Tournament selection
        tournament_size = 3
        
        # Select first parent
        indices1 = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses1 = [fitnesses[i] for i in indices1]
        winner1_idx = indices1[np.argmax(tournament_fitnesses1)]
        parent1 = population[winner1_idx]
        
        # Select second parent
        indices2 = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses2 = [fitnesses[i] for i in indices2]
        winner2_idx = indices2[np.argmax(tournament_fitnesses2)]
        parent2 = population[winner2_idx]
        
        return parent1, parent2
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Args:
            parent1 (list): First parent pattern
            parent2 (list): Second parent pattern
            
        Returns:
            tuple: (child1, child2)
        """
        if np.random.random() > self.crossover_rate:
            return parent1, parent2
            
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, pattern, grid_rows, grid_cols):
        """
        Mutate a pattern.
        
        Args:
            pattern (list): Pattern to mutate
            grid_rows (int): Number of rows in the Template Grid
            grid_cols (int): Number of columns in the Template Grid
            
        Returns:
            list: Mutated pattern
        """
        if np.random.random() > self.mutation_rate:
            return pattern
            
        # Convert to 2D grid
        grid = np.array(pattern).reshape(grid_rows, grid_cols)
        
        # Select a random column
        col = np.random.randint(0, grid_cols)
        
        # Clear the column
        grid[:, col] = 0
        
        # Set a random cell in the column
        row = np.random.randint(0, grid_rows)
        grid[row, col] = 1
        
        return grid.flatten().tolist()
    
    def evolve(self, reference_patterns, grid_rows, grid_cols):
        """
        Evolve a population to find optimal patterns.
        
        Args:
            reference_patterns (list): List of reference patterns
            grid_rows (int): Number of rows in the Template Grid
            grid_cols (int): Number of columns in the Template Grid
            
        Returns:
            tuple: (best_pattern, best_fitness, population)
        """
        # Initialize population
        population = self.initialize_population(grid_rows, grid_cols)
        
        best_pattern = None
        best_fitness = -float('inf')
        
        for generation in range(self.generations):
            # Calculate fitness for each pattern
            fitnesses = [self.fitness(p, reference_patterns) for p in population]
            
            # Find best pattern
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_pattern = population[max_fitness_idx]
                
            logger.info(f"Generation {generation + 1}/{self.generations}, Best fitness: {best_fitness:.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best pattern
            new_population.append(population[max_fitness_idx])
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitnesses)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1, grid_rows, grid_cols)
                child2 = self.mutate(child2, grid_rows, grid_cols)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace old population
            population = new_population
            
        return best_pattern, best_fitness, population


def main():
    """
    Main function to demonstrate the pattern extraction workflow.
    """
    # Set directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'data', 'patterns')
    
    # Initialize pattern extractor
    extractor = PatternExtractor(data_dir, output_dir)
    
    # Extract patterns from 1h timeframe
    timeframe = '1h'
    window_size = 5  # 5 candlesticks per pattern
    stride = 10  # Skip 10 candlesticks between windows
    max_windows = 5000  # Limit to 5000 windows for demonstration
    
    extractor.extract_patterns(
        timeframe, 
        window_size=window_size, 
        stride=stride, 
        max_windows=max_windows,
        grid_rows=10,
        grid_cols=10,
        n_clusters=20
    )
    
    logger.info("Pattern extraction completed successfully")

if __name__ == "__main__":
    main()
