# Python environment
PYTHON = python
VENV = myenv
PIP = $(VENV)/bin/pip

# Scripts
UPDATE_OHLC = scripts.update_ohlc
FIND_CONDITIONS = scripts.find_by_conditions
SHOW_MARKET_DATA = scripts.show_market_data
FETCH_LAST = scripts.fetch_last_candle
FETCH_MARKET = scripts.fetch_market_data

# Default timeframes and tickers
TIMEFRAME ?= 1h
TICKER ?= BTCUSDT
DAYS ?= 7
DEBUG ?= 0

# Convert DEBUG flag to script argument
DEBUG_ARG = $(if $(filter 1 TRUE true,$(DEBUG)),--debug,)

.PHONY: venv install update update-btc update-eth fetch find show last help clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make update        - Update all tickers with latest data"
	@echo "  make fetch        - Fetch historical data from Binance"
	@echo "  make find         - Find patterns with conditions"
	@echo "  make show         - Show market data"
	@echo "  make last         - Show last available candle"
	@echo ""
	@echo "Optional parameters:"
	@echo "  TICKER=BTCUSDT    - Specify ticker (default: BTCUSDT)"
	@echo "  TIMEFRAME=1h      - Specify timeframe (default: 1h)"
	@echo "  DAYS=7           - Number of days to look back"
	@echo "  START=2025-02-01 - Start date for historical data"
	@echo "  END=2025-02-21   - End date for historical data"
	@echo "  PATTERN=Hammer    - Pattern to search for"
	@echo "  RSI_MIN=30       - Minimum RSI value"
	@echo "  RSI_MAX=70       - Maximum RSI value"
	@echo "  DEBUG=1          - Enable debug logging"

	@echo ""
	@echo "Examples:"
	@echo "  make update TICKER=ETHUSDT          - Update ETH with latest data"
	@echo "  make fetch TICKER=BTCUSDT START=2025-02-01 END=2025-02-21  - Fetch historical data"
	@echo "  make find PATTERN=Doji RSI_MIN=65   - Find patterns"
	@echo "  make show TICKER=BTCUSDT DAYS=30    - Show market data"
	@echo "  make last TICKER=BTCUSDT            - Show last candle"
	@echo "  make update DEBUG=1                 - Update with debug logging enabled"

venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

update:
	$(PYTHON) -m $(UPDATE_OHLC) \
		$(if $(TICKER),--ticker $(TICKER)) \
		$(if $(TIMEFRAME),--timeframe $(TIMEFRAME)) \
		$(DEBUG_ARG)

find:
	$(PYTHON) -m $(FIND_CONDITIONS) \
		$(if $(PATTERN),--pattern $(PATTERN)) \
		$(if $(TICKER),--ticker $(TICKER)) \
		$(if $(TIMEFRAME),--timeframe $(TIMEFRAME)) \
		$(if $(DAYS),--days $(DAYS)) \
		$(if $(RSI_MIN),--rsi-min $(RSI_MIN)) \
		$(if $(RSI_MAX),--rsi-max $(RSI_MAX)) \
		$(DEBUG_ARG)

show:
	$(PYTHON) -m $(SHOW_MARKET_DATA) \
		$(if $(TICKER),--ticker $(TICKER)) \
		$(if $(TIMEFRAME),--timeframe $(TIMEFRAME)) \
		$(if $(DAYS),--days $(DAYS)) \
		$(DEBUG_ARG)

last:
	$(PYTHON) -m $(FETCH_LAST) \
		$(if $(TICKER),--ticker $(TICKER)) \
		$(if $(TIMEFRAME),--timeframe $(TIMEFRAME)) \
		$(DEBUG_ARG)

fetch:
	$(PYTHON) -m $(FETCH_MARKET) \
		$(if $(TICKER),--ticker $(TICKER)) \
		$(if $(TIMEFRAME),--timeframe $(TIMEFRAME)) \
		$(if $(START),--start $(START)) \
		$(if $(END),--end $(END)) \
		$(if $(DAYS),--days $(DAYS)) \
		$(DEBUG_ARG)

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete 