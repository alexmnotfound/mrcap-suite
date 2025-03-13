# Python environment
PYTHON = python
VENV = myenv
PIP = $(VENV)/bin/pip

# Scripts
UPDATE_OHLC = scripts.update_ohlc
GET_BY_CONDITIONS = scripts.get_by_conditions
GET_MARKET = scripts.get_market_data
GET_LAST = scripts.get_last_candle
FETCH_MARKET = scripts.fetch_market_data
GET_ANALYSIS = scripts.get_analysis

# Default values
TICKER ?= BTCUSDT
DEBUG ?= FALSE

# Convert DEBUG flag to script argument
DEBUG_ARG = $(if $(filter TRUE true,$(DEBUG)),--debug,)

.PHONY: venv install update fetch find show last analysis help clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make update        - Update all tickers with latest data"
	@echo "  make fetch         - Fetch historical data"
	@echo "  make find          - Find patterns with conditions"
	@echo "  make show          - Show market data"
	@echo "  make last          - Show last available candle"
	@echo "  make analysis      - Get market analysis with indicators"
	@echo ""
	@echo "Optional parameters:"
	@echo "  TICKER=BTCUSDT    - Specify ticker (default: BTCUSDT)"
	@echo "  TIMEFRAME=1h      - Specify timeframe (1h/4H/1D)"
	@echo "  DAYS=7           - Number of days to look back"
	@echo "  START=2025-02-01 - Start date for historical data"
	@echo "  END=2025-02-21   - End date for historical data"
	@echo "  PATTERN=Hammer    - Pattern to search for"
	@echo "  RSI_MIN=30       - Minimum RSI value"
	@echo "  RSI_MAX=70       - Maximum RSI value"
	@echo "  VOLUME_MIN=1000  - Minimum volume"
	@echo "  OUTPUT=data.csv   - Output file for analysis"
	@echo "  DEBUG=TRUE       - Enable debug logging"

	@echo ""
	@echo "Examples:"
	@echo "  make update TICKER=ETHUSDT          - Update ETH with latest data"
	@echo "  make fetch TICKER=BTCUSDT START=2025-02-01 END=2025-02-21  - Fetch historical data"
	@echo "  make find PATTERN=Doji RSI_MIN=65   - Find patterns"
	@echo "  make show TICKER=BTCUSDT DAYS=30    - Show market data"
	@echo "  make last TICKER=BTCUSDT            - Show last candle"
	@echo "  make analysis TICKER=BTCUSDT DAYS=7 OUTPUT=btc_analysis.csv - Get BTC analysis"
	@echo "  make update DEBUG=TRUE              - Update with debug logging enabled"

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
	$(PYTHON) -m $(GET_BY_CONDITIONS) \
		$(if $(PATTERN),--pattern $(PATTERN)) \
		$(if $(TICKER),--ticker $(TICKER)) \
		$(if $(TIMEFRAME),--timeframe $(TIMEFRAME)) \
		$(if $(DAYS),--days $(DAYS)) \
		$(if $(RSI_MIN),--rsi-min $(RSI_MIN)) \
		$(if $(RSI_MAX),--rsi-max $(RSI_MAX)) \
		$(if $(VOLUME_MIN),--volume-min $(VOLUME_MIN)) \
		$(DEBUG_ARG)

show:
	$(PYTHON) -m $(GET_MARKET) \
		$(if $(TICKER),--ticker $(TICKER)) \
		$(if $(TIMEFRAME),--timeframe $(TIMEFRAME)) \
		$(if $(DATE),--date $(DATE)) \
		$(if $(START),--start $(START)) \
		$(if $(END),--end $(END)) \
		$(DEBUG_ARG)

last:
	$(PYTHON) -m $(GET_LAST) \
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
		$(if $(INIT_DB),--init-db) \
		$(DEBUG_ARG)

analysis:
	$(PYTHON) -m $(GET_ANALYSIS) \
		$(if $(TICKER),--ticker $(TICKER)) \
		$(if $(TIMEFRAME),--timeframe $(TIMEFRAME)) \
		$(if $(DAYS),--days $(DAYS)) \
		$(if $(OUTPUT),--output $(OUTPUT)) \
		$(DEBUG_ARG)

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete 