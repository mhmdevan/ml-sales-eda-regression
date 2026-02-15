APP ?= sales
HOST ?= 127.0.0.1

.PHONY: train serve help

help:
	@echo "Usage:"
	@echo "  make train APP=sales"
	@echo "  make train APP=housing"
	@echo "  make serve APP=sales"
	@echo "  make serve APP=housing"

train:
	@if [ "$(APP)" = "sales" ]; then \
		python -m sales_forecasting_regression.train; \
	elif [ "$(APP)" = "housing" ] || [ "$(APP)" = "california" ]; then \
		python -m california_housing_template.train; \
	else \
		echo "Unsupported APP=$(APP). Use APP=sales or APP=housing"; \
		exit 1; \
	fi

serve:
	@if [ "$(APP)" = "sales" ]; then \
		uvicorn sales_forecasting_regression.api:app --host $(HOST) --port 8000 --reload; \
	elif [ "$(APP)" = "housing" ] || [ "$(APP)" = "california" ]; then \
		uvicorn california_housing_template.api:app --host $(HOST) --port 8001 --reload; \
	else \
		echo "Unsupported APP=$(APP). Use APP=sales or APP=housing"; \
		exit 1; \
	fi
