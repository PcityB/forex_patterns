# Forex Pattern Discovery Framework - User Guide

This comprehensive guide will help you navigate and utilize the Forex Pattern Discovery Framework to identify novel, statistically significant, and profitable candlestick patterns in forex market data.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Management](#data-management)
3. [Pattern Extraction](#pattern-extraction)
4. [Pattern Analysis](#pattern-analysis)
5. [Visualization and Backtesting](#visualization-and-backtesting)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forex-pattern-discovery.git
cd forex-pattern-discovery
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python src/main.py
```

5. Access the web interface at http://localhost:5000

### System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for larger datasets)
- Modern web browser (Chrome, Firefox, Edge)
- GPU support (optional, but recommended for faster pattern extraction)

## Data Management

### Supported Data Formats

The framework accepts forex data in CSV format with the following columns:
- Date/Time (required)
- Open price (required)
- High price (required)
- Low price (required)
- Close price (required)
- Volume (optional)

Example CSV format:
```
datetime,open,high,low,close,volume
2023-01-01 00:00:00,1.2345,1.2350,1.2340,1.2348,1000
2023-01-01 00:15:00,1.2348,1.2355,1.2345,1.2352,1200
```

### Uploading Data

1. Navigate to the "Data Management" section from the main dashboard
2. Click "Upload Data" button
3. Select your CSV file
4. Choose the appropriate timeframe (e.g., 15m, 1h, 4h)
5. Click "Upload" to begin the process

### Data Preprocessing

After uploading, you can configure preprocessing options:

1. **Normalization**: Standardize price data for better pattern recognition
   - Z-score normalization
   - Min-max scaling
   - Percentage change

2. **Feature Engineering**: Generate additional technical indicators
   - Moving averages (SMA, EMA)
   - RSI, MACD, Bollinger Bands
   - Candlestick features (body size, shadow ratios)

3. **Data Filtering**:
   - Remove outliers
   - Handle missing values
   - Select date ranges

4. Click "Process Data" to apply your selected options

### Data Preview

Before proceeding to pattern extraction, you can preview the processed data:

1. View summary statistics
2. Check for missing values or anomalies
3. Visualize price action and indicators
4. Confirm data quality and completeness

## Pattern Extraction

### Configuring Extraction Parameters

1. Navigate to the "Pattern Extraction" section
2. Select the preprocessed dataset
3. Configure extraction parameters:
   - Window size (number of candlesticks per pattern)
   - Overlap percentage
   - Similarity threshold for clustering
   - Number of clusters (or auto-detection)

### Running Pattern Extraction

1. Click "Extract Patterns" to begin the process
2. The system will:
   - Apply the Template Grid approach to standardize patterns
   - Calculate Dynamic Time Warping distances between patterns
   - Perform hierarchical clustering to group similar patterns
   - Generate Pattern Identification Codes (PICs)

3. This process may take several minutes depending on:
   - Dataset size
   - Window size
   - Clustering parameters
   - Available computational resources

### Viewing Extracted Patterns

Once extraction is complete, you can:

1. View pattern distribution across clusters
2. Examine representative patterns from each cluster
3. Explore pattern characteristics and metadata
4. Select patterns for further analysis

## Pattern Analysis

### Statistical Validation

1. Navigate to the "Pattern Analysis" section
2. Select the extracted patterns to analyze
3. Configure analysis parameters:
   - Lookahead periods for profitability calculation
   - Statistical significance threshold (p-value)
   - Minimum pattern occurrence threshold

4. Click "Analyze Patterns" to begin statistical validation

### Interpreting Analysis Results

The analysis provides several key metrics:

1. **Profitability Metrics**:
   - Average return per pattern
   - Win rate (percentage of profitable occurrences)
   - Profit factor (gross profits / gross losses)
   - Maximum consecutive wins/losses

2. **Statistical Significance**:
   - P-values for each pattern cluster
   - Confidence intervals
   - Multiple testing correction results

3. **Pattern Characteristics**:
   - Frequency of occurrence
   - Market conditions correlation
   - Temporal distribution

## Visualization and Backtesting

### Interactive Visualizations

1. Navigate to the "Visualization" section
2. Select analysis results to visualize
3. Explore interactive charts:
   - Cluster profitability comparison
   - Statistical significance visualization
   - Pattern distribution charts
   - Feature importance plots

### Backtesting Trading Strategies

1. Navigate to the "Backtesting" section
2. Configure backtest parameters:
   - Initial capital
   - Position sizing rules
   - Stop-loss and take-profit levels
   - Risk management settings
   - Transaction costs and slippage

3. Select profitable pattern clusters to include in the strategy
4. Click "Run Backtest" to simulate trading performance

### Analyzing Backtest Results

The backtest results provide comprehensive performance metrics:

1. **Performance Metrics**:
   - Total return
   - Sharpe ratio
   - Maximum drawdown
   - Win rate and profit factor

2. **Equity Curve**: Visual representation of account growth over time

3. **Trade Statistics**:
   - Number of trades
   - Average win/loss
   - Holding periods
   - Monthly returns

## Advanced Configuration

### Customizing Pattern Recognition

Advanced users can modify the pattern recognition algorithms:

1. **Template Grid Configuration**:
   - Grid resolution
   - Normalization methods
   - Feature weighting

2. **Clustering Parameters**:
   - Distance metrics
   - Linkage methods
   - Cluster validation techniques

3. **Machine Learning Integration**:
   - Feature selection
   - Model selection and hyperparameter tuning
   - Ensemble methods

### System Configuration

Optimize system performance based on your hardware:

1. **Computational Resources**:
   - CPU thread allocation
   - Memory usage limits
   - GPU acceleration settings

2. **Storage Management**:
   - Data retention policies
   - Caching strategies
   - Export/import configurations

## Troubleshooting

### Common Issues and Solutions

1. **Data Import Problems**:
   - Ensure CSV format matches expected structure
   - Check for date/time format consistency
   - Verify no missing required columns

2. **Performance Issues**:
   - Reduce dataset size for initial testing
   - Decrease pattern window size
   - Limit number of technical indicators
   - Enable GPU acceleration if available

3. **Pattern Extraction Failures**:
   - Check preprocessing results for anomalies
   - Adjust similarity thresholds
   - Increase minimum pattern occurrence threshold

4. **Visualization Errors**:
   - Clear browser cache
   - Update to latest browser version
   - Check console for JavaScript errors

### Getting Support

If you encounter issues not covered in this guide:

1. Check the GitHub repository issues section
2. Review documentation for updates
3. Contact the development team with:
   - Detailed description of the issue
   - Steps to reproduce
   - System specifications
   - Error logs (located in the logs directory)

---

This user guide provides a comprehensive overview of the Forex Pattern Discovery Framework. For more detailed information on specific components, please refer to the technical documentation.
