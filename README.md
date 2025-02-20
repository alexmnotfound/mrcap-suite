# MrCap Suite
A comprehensive market data and technical analysis toolkit for cryptocurrency trading.

## 🚧 Work in Progress
This project is under active development. The core infrastructure is being built to support various trading applications.

## Overview
MrCap Suite provides tools for:
- Fetching and storing cryptocurrency market data
- Calculating technical indicators
- Building trading strategies and applications

## Prerequisites
- Python 3.8+
- PostgreSQL 12+

## Database Setup
1. Install PostgreSQL:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```
2. Create a database:
```bash
sudo -u postgres psql
CREATE DATABASE mrcap_db;
```
3. Configure database connection:
- Copy libs/market_data/config.py_sample to libs/market_data/config.py
- Update the configuration with your database credentials

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/mrcap-suite.git
cd mrcap-suite
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Initialize the database:
```bash
python -m libs.market_data.init_db
```

## Usage
1. Fetch Market Data
```bash
python scripts/run_market_data.py
```
2. View Market Data
```bash
python scripts/show_market_data.py
```
3. Run Analysis
```bash
python scripts/fetch_analysis.py
```

## Project Structure
```bash
mrcap-suite/
├── libs/
│   ├── market_data/    # Market data management
│   ├── ta_lib/         # Technical analysis library
│   └── utils/          # Utility functions
├── scripts/            # Command-line tools
└── tests/              # Test suite
```
Planned Features
[ ] Real-time market data streaming
[ ] Web interface for data visualization
[ ] Trading strategy backtesting engine
[ ] Portfolio management tools
[ ] Risk management system
[ ] Automated trading bots

## Contributing
This project is currently in development. Feel free to open issues for bugs or feature requests.

## Disclaimer
This software is for educational purposes only. Use at your own risk. The authors and contributors are not responsible for any financial losses incurred through the use of this software.
