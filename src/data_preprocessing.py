"""
Data Preprocessing Module for Forex Pattern Discovery Framework
This module handles loading, cleaning, and preprocessing of forex data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_preprocessing')

class ForexDataPreprocessor:
    """
    Class for preprocessing forex data from CSV files.
    """
    
    def __init__(self, data_dir):
        """
        Initialize the preprocessor with the data directory.
        
        Args:
            data_dir (str): Directory containing the forex data files
        """
        self.data_dir = data_dir
        self.raw_data = {}
        self.processed_data = {}
        self.scalers = {}

    def load_data(self, filename, delimiter=None):
        """
        Load a single data file directly by filename.
        
        Args:
            filename (str): Filename of the CSV file to load.
        
        Returns:
            dict: Dictionary with a single loaded dataframe keyed by timeframe extracted from filename.
        """
        if delimiter is None:
            # Use your existing detection logic or default semicolon
            delimiter = ';'  # Your current default
        
        logger.info(f"Loading data file {filename} with delimiter '{delimiter}'")
        file_path = os.path.join(self.data_dir, filename)
        try:
            # Load data with semicolon delimiter
            df = pd.read_csv(file_path, delimiter=delimiter)
            # Extract timeframe from filename, assuming format like XAU_{timeframe}_data.csv
            if filename.startswith("XAU_") and filename.endswith("_data.csv"):
                tf = filename.replace("XAU_", "").replace("_data.csv", "")
            else:
                tf = "custom"
            self.raw_data[tf] = df
            logger.info(f"Loaded {tf} data with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            self.raw_data = {}
        
        return self.raw_data
    
    def clean_data(self, timeframe=None):
        """
        Clean the loaded data by handling missing values, duplicates, and converting data types.
        
        Args:
            timeframe (str, optional): Specific timeframe to clean. If None, cleans all loaded timeframes.
            
        Returns:
            dict: Dictionary of cleaned dataframes by timeframe
        """
        logger.info("Cleaning data")
        
        timeframes = [timeframe] if timeframe else list(self.raw_data.keys())
        
        for tf in timeframes:
            if tf not in self.raw_data:
                logger.warning(f"No data loaded for timeframe {tf}")
                continue
                
            df = self.raw_data[tf].copy()
            
            # Convert date column to datetime
            logger.info(f"Converting date format for {tf} data")
            for i in tqdm(range(len(df)), desc=f"Converting dates for {tf}"):
                try:
                    df.iloc[i, df.columns.get_loc('Date')] = pd.to_datetime(df.iloc[i, df.columns.get_loc('Date')], format='%Y.%m.%d %H:%M')
                except Exception as e:
                    logger.warning(f"Error converting date at index {i}: {e}. Trying alternative format.")
                    try:
                        df.iloc[i, df.columns.get_loc('Date')] = pd.to_datetime(df.iloc[i, df.columns.get_loc('Date')])
                    except Exception as e:
                        logger.error(f"Failed to convert date at index {i}: {e}")
            
            # Set Date as index
            df.set_index('Date', inplace=True)
            
            # Handle missing values
            logger.info(f"Handling missing values for {tf} data")
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                logger.info(f"Found {missing_values.sum()} missing values")
                # Forward fill for missing values
                df.fillna(method='ffill', inplace=True)
                # If any remaining, use backward fill
                df.fillna(method='bfill', inplace=True)
            
            # Remove duplicates
            logger.info(f"Removing duplicates for {tf} data")
            initial_rows = len(df)
            df = df[~df.index.duplicated(keep='first')]
            if len(df) < initial_rows:
                logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
            
            # Ensure all price columns are numeric
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle any remaining NaN values after conversion
            if df.isnull().sum().sum() > 0:
                logger.warning(f"Found {df.isnull().sum().sum()} NaN values after type conversion")
                df.dropna(inplace=True)
                logger.info(f"Dropped rows with NaN values, {len(df)} rows remaining")
            
            # Sort by date
            df.sort_index(inplace=True)
            
            self.processed_data[tf] = df
            logger.info(f"Cleaned {tf} data, final shape: {df.shape}")
    
    def normalize_data(self, timeframe=None):
        """
        Normalize the data using Min-Max scaling.
        
        Args:
            timeframe (str, optional): Specific timeframe to normalize. If None, normalizes all processed timeframes.
            
        Returns:
            dict: Dictionary of normalized dataframes by timeframe
        """
        logger.info("Normalizing data")
        
        timeframes = [timeframe] if timeframe else list(self.processed_data.keys())
        
        for tf in timeframes:
            if tf not in self.processed_data:
                logger.warning(f"No processed data for timeframe {tf}")
                continue
                
            df = self.processed_data[tf].copy()
            
            # Select columns to normalize (price and derived features, not categorical or binary)
            price_cols = ['Open', 'High', 'Low', 'Close', 'body_size', 'upper_shadow', 'lower_shadow']
            derived_cols = [col for col in df.columns if any(x in col for x in ['ma_', 'volatility_', 'range_', 'atr_', 'rsi_', 'macd'])]
            cols_to_normalize = price_cols + derived_cols
            
            # Create and fit scaler
            scaler = MinMaxScaler()
            df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
            
            # Store scaler for inverse transformation if needed
            self.scalers[tf] = scaler
            
            self.processed_data[tf] = df
            logger.info(f"Normalized {tf} data")
            
        return self.processed_data

    def engineer_features(self, timeframe=None):
        """
        Engineer additional features from the cleaned data.
        
        Args:
            timeframe (str, optional): Specific timeframe to process. If None, processes all cleaned timeframes.
            
        Returns:
            dict: Dictionary of dataframes with engineered features by timeframe
        """
        logger.info("Engineering features")
        
        timeframes = [timeframe] if timeframe else list(self.processed_data.keys())
        
        for tf in timeframes:
            if tf not in self.processed_data:
                logger.warning(f"No cleaned data for timeframe {tf}")
                continue
                
            df = self.processed_data[tf].copy()
            
            # Calculate basic candlestick features
            logger.info(f"Calculating candlestick features for {tf} data")
            
            # Body size and direction
            df['body_size'] = abs(df['Close'] - df['Open'])
            df['body_ratio'] = df['body_size'] / (df['High'] - df['Low'])
            df['direction'] = np.where(df['Close'] >= df['Open'], 1, -1)  # 1 for bullish, -1 for bearish
            
            # Upper and lower shadows
            df['upper_shadow'] = df.apply(
                lambda x: x['High'] - x['Close'] if x['Close'] >= x['Open'] else x['High'] - x['Open'],
                axis=1
            )
            df['lower_shadow'] = df.apply(
                lambda x: x['Open'] - x['Low'] if x['Close'] >= x['Open'] else x['Close'] - x['Low'],
                axis=1
            )
            df['upper_shadow_ratio'] = df['upper_shadow'] / (df['High'] - df['Low'])
            df['lower_shadow_ratio'] = df['lower_shadow'] / (df['High'] - df['Low'])
            
            # Calculate price movements
            df['price_change'] = df['Close'].diff()
            df['price_pct_change'] = df['Close'].pct_change() * 100
            
            # Calculate rolling statistics
            window_sizes = [5, 10, 20]
            for window in tqdm(window_sizes, desc=f"Calculating rolling statistics for {tf}"):
                # Moving averages
                df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
                # Volatility (standard deviation)
                df[f'volatility_{window}'] = df['Close'].rolling(window=window).std()
                # Price range
                df[f'range_{window}'] = df['High'].rolling(window=window).max() - df['Low'].rolling(window=window).min()
                    
            # Calculate RSI (Relative Strength Index)
            logger.info(f"Calculating RSI for {tf} data")
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            # Calculate Relative Strength (RS)
            rs = gain / loss

            # Calculate RSI using the standard formula
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # Handle division by zero cases (when loss is 0)
            df['rsi_14'] = df['rsi_14'].fillna(100)  # RSI = 100 when no losses

            # MACD (Moving Average Convergence Divergence
            # MACD (Moving Average Convergence Divergence)
            logger.info(f"Calculating MACD for {tf} data")
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # ATR (Average True Range)
            logger.info(f"Calculating ATR for {tf} data")
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = df['true_range'].rolling(14).mean()
            
            # Drop rows with NaN values resulting from calculations
            initial_rows = len(df)
            df.dropna(inplace=True)
            if len(df) < initial_rows:
                logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values after feature engineering")
            
            self.processed_data[tf] = df
            logger.info(f"Engineered features for {tf} data, final shape: {df.shape}")
    
    def prepare_pattern_data(self, timeframe, window_size=10):
        """
        Prepare data for pattern extraction by creating sliding windows.
        
        Args:
            timeframe (str): Timeframe to prepare data for
            window_size (int): Size of the sliding window for pattern extraction
            
        Returns:
            tuple: (X, timestamps) where X is the windowed data and timestamps are the corresponding end times
        """
        if timeframe not in self.processed_data:
            logger.error(f"No processed data for timeframe {timeframe}")
            return None, None
            
        df = self.processed_data[timeframe]
        
        # Select features for pattern extraction
        pattern_features = [
            'Open', 'High', 'Low', 'Close', 
            'body_size', 'body_ratio', 'direction',
            'upper_shadow_ratio', 'lower_shadow_ratio'
        ]
        
        # Ensure all required features are available
        available_features = [f for f in pattern_features if f in df.columns]
        if len(available_features) < len(pattern_features):
            missing = set(pattern_features) - set(available_features)
            logger.warning(f"Missing features for pattern extraction: {missing}")
        
        # Create sliding windows
        X = []
        timestamps = []
        
        for i in tqdm(range(len(df) - window_size + 1), desc=f"Creating sliding windows for {timeframe}"):
            window = df.iloc[i:i+window_size][available_features].values
            X.append(window)
            timestamps.append(df.index[i+window_size-1])
        
        X = np.array(X)
        logger.info(f"Prepared {len(X)} pattern windows of size {window_size} for {timeframe} data")
        
        return X, timestamps
    
    def save_processed_data(self, output_dir=None):
        """
        Save processed data to CSV files.
        
        Args:
            output_dir (str, optional): Directory to save processed data. If None, uses data_dir/processed.
            
        Returns:
            dict: Dictionary of saved file paths by timeframe
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'processed')
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving processed data to {output_dir}")
        
        saved_files = {}
        for tf, df in self.processed_data.items():
            output_file = os.path.join(output_dir, f"XAU_{tf}_processed.csv")
            df.to_csv(output_file)
            saved_files[tf] = output_file
            logger.info(f"Saved {tf} data to {output_file}")
            
        return saved_files
    
    def visualize_data(self, timeframe, n_samples=1000, output_dir=None):
        """
        Create visualizations of the processed data.
        
        Args:
            timeframe (str): Timeframe to visualize
            n_samples (int): Number of samples to visualize
            output_dir (str, optional): Directory to save visualizations. If None, uses data_dir/visualizations.
            
        Returns:
            list: List of saved visualization file paths
        """
        if timeframe not in self.processed_data:
            logger.error(f"No processed data for timeframe {timeframe}")
            return []
            
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'visualizations')
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Creating visualizations for {timeframe} data")
        
        df = self.processed_data[timeframe]
        
        # Use the last n_samples for visualization
        if len(df) > n_samples:
            df = df.iloc[-n_samples:]
        
        saved_files = []
        
        # Candlestick chart
        plt.figure(figsize=(12, 6))
        plt.title(f'XAUUSD {timeframe} Candlestick Chart')
        
        # Plot candlesticks
        for i in tqdm(range(len(df)), desc=f"Plotting candlesticks for {timeframe}"):
            # Get the data point
            date = df.index[i]
            op, high, low, close = df.iloc[i][['Open', 'High', 'Low', 'Close']]
            
            # Determine if bullish or bearish
            if close >= op:
                color = 'green'
            else:
                color = 'red'
            
            # Plot the body
            plt.plot([date, date], [op, close], color=color, linewidth=2)
            
            # Plot the wicks
            plt.plot([date, date], [low, high], color='black', linewidth=1)
        
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        candlestick_file = os.path.join(output_dir, f"XAU_{timeframe}_candlestick.png")
        plt.savefig(candlestick_file)
        plt.close()
        saved_files.append(candlestick_file)
        logger.info(f"Saved candlestick chart to {candlestick_file}")
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = df.corr()
        plt.imshow(corr, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title(f'XAUUSD {timeframe} Feature Correlation')
        plt.tight_layout()
        
        # Save the figure
        correlation_file = os.path.join(output_dir, f"XAU_{timeframe}_correlation.png")
        plt.savefig(correlation_file)
        plt.close()
        saved_files.append(correlation_file)
        logger.info(f"Saved correlation heatmap to {correlation_file}")
        
        return saved_files

def main(filename=None):
    """
    Main function to demonstrate the data preprocessing workflow.
    Accepts an optional filename to process.
    """
    # Set data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Initialize preprocessor
    preprocessor = ForexDataPreprocessor(data_dir)
    
    # Use provided filename or default example
    if filename is None:
        filename = "XAU_1h_data.csv"
    
    # Load data file directly
    preprocessor.load_data(filename)
    
    # Clean data
    preprocessor.clean_data()
    
    # Engineer features
    preprocessor.engineer_features()
    
    # Normalize data
    preprocessor.normalize_data()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Optionally create visualizations for all loaded timeframes
    for tf in preprocessor.processed_data.keys():
        preprocessor.visualize_data(tf)
    
    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()
