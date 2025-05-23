# Forex Pattern Discovery Framework

A robust, data-driven web application for identifying and validating novel candlestick patterns in forex market data.

## Overview

This framework provides a comprehensive solution for discovering previously unrecognized, statistically significant, and profitable candlestick patterns in high-frequency intraday forex market data. It extends beyond conventionally acknowledged patterns to provide actionable trading insights through advanced pattern recognition and machine learning techniques.

## Features

- **Data Management**: Upload, preprocess, and manage forex data across multiple timeframes
- **Pattern Extraction**: Discover novel patterns using Template Grid approach and Dynamic Time Warping
- **Pattern Analysis**: Validate patterns with statistical significance testing and profitability metrics
- **Interactive Visualization**: Explore patterns through interactive charts and visualizations
- **Backtesting**: Test trading strategies based on discovered patterns with realistic market conditions

## Architecture

The application is built with a modular architecture consisting of five main layers:

1. **Data Management Layer**: Handles data ingestion, preprocessing, and storage
2. **Pattern Discovery Layer**: Implements Template Grid system and pattern extraction algorithms
3. **Pattern Validation Layer**: Performs statistical analysis and backtesting
4. **Presentation Layer**: Provides web interface and visualization tools
5. **Infrastructure Layer**: Supports cloud computing with GPU acceleration

## Installation

### Prerequisites

- Python 3.8+
- Flask
- Pandas, NumPy, Matplotlib
- Scikit-learn, TensorFlow (for machine learning components)
- Plotly (for interactive visualizations)

### Setup

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

## Usage

### Data Management

1. Upload forex data in CSV format
2. Configure preprocessing options (normalization, feature engineering)
3. Preview and validate processed data

### Pattern Extraction

1. Select timeframe and configure extraction parameters
2. Run pattern extraction process
3. View extracted patterns and their distribution

### Pattern Analysis

1. Select patterns for analysis
2. Configure analysis parameters (lookahead periods, significance threshold)
3. View statistical significance and profitability metrics

### Visualization and Backtesting

1. Explore interactive visualizations of pattern performance
2. Configure and run backtests with realistic trading conditions
3. Analyze backtest results and optimize strategies

## Project Structure

```
forex_pattern_app/
├── data/                      # Data storage
│   ├── raw/                   # Raw forex data files
│   ├── processed/             # Preprocessed data
│   ├── patterns/              # Extracted patterns
│   └── analysis/              # Analysis results
├── src/                       # Source code
│   ├── main.py                # Application entry point
│   ├── routes/                # Flask route definitions
│   │   ├── data_routes.py     # Data management routes
│   │   ├── patterns_routes.py # Pattern extraction routes
│   │   ├── analysis_routes.py # Analysis routes
│   │   └── visualization_routes.py # Visualization routes
│   ├── static/                # Static assets
│   │   ├── css/               # Stylesheets
│   │   ├── js/                # JavaScript files
│   │   └── visualization/     # Generated visualizations
│   ├── templates/             # HTML templates
│   │   ├── data/              # Data management templates
│   │   ├── patterns/          # Pattern extraction templates
│   │   ├── analysis/          # Analysis templates
│   │   └── visualization/     # Visualization templates
│   └── modules/               # Core functionality modules
│       ├── data_preprocessing.py # Data preprocessing
│       ├── pattern_extraction.py # Pattern extraction
│       └── pattern_analysis.py   # Pattern analysis
└── requirements.txt           # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed to address the need for more sophisticated pattern discovery in forex trading
- Special thanks to the open-source community for providing the tools and libraries that made this possible
