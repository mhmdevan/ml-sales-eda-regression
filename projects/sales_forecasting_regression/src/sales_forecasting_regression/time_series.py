from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sales_forecasting_regression.config import SalesPaths
from sales_forecasting_regression.data import SalesDatasetLoader


def _monthly_sales_series(
    *,
    frame: pd.DataFrame,
    product_line: str,
    country: str,
) -> pd.Series:
    scoped = frame.copy()
    if "PRODUCTLINE" in scoped.columns:
        scoped = scoped.loc[scoped["PRODUCTLINE"] == product_line]
    if "COUNTRY" in scoped.columns:
        scoped = scoped.loc[scoped["COUNTRY"] == country]
    if scoped.empty:
        return pd.Series(dtype=float, name="SALES")

    if "ORDERDATE" in scoped.columns:
        dates = pd.to_datetime(scoped["ORDERDATE"], errors="coerce")
    elif {"YEAR_ID", "MONTH_ID"}.issubset(set(scoped.columns)):
        dates = pd.to_datetime(
            {
                "year": pd.to_numeric(scoped["YEAR_ID"], errors="coerce"),
                "month": pd.to_numeric(scoped["MONTH_ID"], errors="coerce"),
                "day": 1,
            },
            errors="coerce",
        )
    else:
        return pd.Series(dtype=float, name="SALES")

    scoped = scoped.assign(_period=dates.dt.to_period("M"))
    scoped = scoped.dropna(subset=["_period", "SALES"])
    if scoped.empty:
        return pd.Series(dtype=float, name="SALES")

    aggregated = scoped.groupby("_period", as_index=True)["SALES"].sum().sort_index()
    period_range = pd.period_range(start=aggregated.index.min(), end=aggregated.index.max(), freq="M")
    aggregated = aggregated.reindex(period_range, fill_value=0.0)
    series = aggregated.to_timestamp(how="start")
    series.name = "SALES"
    return series.astype(float)


def _evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    eps = 1e-8
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), eps))) * 100.0)
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def _naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    if train.empty:
        return np.zeros(horizon, dtype=float)
    return np.repeat(float(train.iloc[-1]), horizon).astype(float)


def _fit_sarima_forecast(train: pd.Series, horizon: int) -> dict[str, Any]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return {
            "available": False,
            "reason": "statsmodels is not installed",
            "forecast": [],
            "aic": None,
            "order": None,
            "seasonal_order": None,
        }

    if train.shape[0] < 12:
        return {
            "available": False,
            "reason": "insufficient history for sarima",
            "forecast": [],
            "aic": None,
            "order": None,
            "seasonal_order": None,
        }

    orders: list[tuple[int, int, int]] = [(1, 1, 1), (1, 0, 1), (2, 1, 1)]
    seasonal_orders: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)]
    if train.shape[0] >= 24:
        seasonal_orders.extend([(1, 1, 0, 12), (0, 1, 1, 12)])

    best: dict[str, Any] | None = None
    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                model = SARIMAX(
                    train.astype(float),
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_result = model.fit(disp=False)
                aic = float(fit_result.aic)
                if not math.isfinite(aic):
                    continue
                if best is None or aic < best["aic"]:
                    forecast = fit_result.get_forecast(steps=horizon).predicted_mean.to_numpy(dtype=float)
                    best = {
                        "available": True,
                        "reason": None,
                        "forecast": [float(value) for value in forecast],
                        "aic": aic,
                        "order": list(order),
                        "seasonal_order": list(seasonal_order),
                    }
            except Exception:
                continue

    if best is None:
        return {
            "available": False,
            "reason": "sarima fitting failed",
            "forecast": [],
            "aic": None,
            "order": None,
            "seasonal_order": None,
        }
    return best


def _garch_risk_summary(returns: pd.Series) -> dict[str, Any]:
    if returns.empty:
        return {
            "available": False,
            "reason": "insufficient returns history",
            "method": None,
            "next_period_volatility": None,
            "volatility_annualized": None,
            "var_95": None,
            "cvar_95": None,
            "return_mean": None,
            "observations": 0,
        }

    returns = returns.astype(float)
    var_95 = float(np.quantile(returns, 0.05))
    tail = returns[returns <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95
    volatility_annualized = float(returns.std(ddof=0) * np.sqrt(12))
    return_mean = float(returns.mean())

    try:
        from arch import arch_model
    except Exception:
        ewma = returns.ewm(alpha=0.06).std(bias=False).dropna()
        next_vol = float(ewma.iloc[-1]) if not ewma.empty else float(returns.std(ddof=0))
        return {
            "available": True,
            "reason": "arch is not installed; used ewma fallback",
            "method": "ewma_fallback",
            "next_period_volatility": next_vol,
            "volatility_annualized": volatility_annualized,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "return_mean": return_mean,
            "observations": int(returns.shape[0]),
        }

    if returns.shape[0] < 24:
        ewma = returns.ewm(alpha=0.06).std(bias=False).dropna()
        next_vol = float(ewma.iloc[-1]) if not ewma.empty else float(returns.std(ddof=0))
        return {
            "available": True,
            "reason": "insufficient observations for stable garch fit; used ewma fallback",
            "method": "ewma_fallback",
            "next_period_volatility": next_vol,
            "volatility_annualized": volatility_annualized,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "return_mean": return_mean,
            "observations": int(returns.shape[0]),
        }

    try:
        fitted = arch_model(returns * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        variance_forecast = float(fitted.forecast(horizon=1).variance.iloc[-1, 0])
        next_volatility = max(0.0, np.sqrt(variance_forecast) / 100.0)
        return {
            "available": True,
            "reason": None,
            "method": "garch_1_1",
            "next_period_volatility": float(next_volatility),
            "volatility_annualized": volatility_annualized,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "return_mean": return_mean,
            "observations": int(returns.shape[0]),
        }
    except Exception as exc:
        ewma = returns.ewm(alpha=0.06).std(bias=False).dropna()
        next_vol = float(ewma.iloc[-1]) if not ewma.empty else float(returns.std(ddof=0))
        return {
            "available": True,
            "reason": f"garch fitting failed; used ewma fallback: {exc}",
            "method": "ewma_fallback",
            "next_period_volatility": next_vol,
            "volatility_annualized": volatility_annualized,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "return_mean": return_mean,
            "observations": int(returns.shape[0]),
        }


def generate_segment_time_series_report(
    *,
    frame: pd.DataFrame,
    output_path: Path,
    product_line: str = "Classic Cars",
    country: str = "USA",
    holdout_periods: int = 6,
    forecast_horizon: int = 3,
) -> dict[str, Any]:
    if holdout_periods < 1:
        raise ValueError("holdout_periods must be at least 1.")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1.")

    series = _monthly_sales_series(frame=frame, product_line=product_line, country=country)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "segment": {
            "product_line": product_line,
            "country": country,
        },
        "series_points": int(series.shape[0]),
        "series_start": str(series.index.min().date()) if not series.empty else None,
        "series_end": str(series.index.max().date()) if not series.empty else None,
        "models": {},
        "selection": {},
        "forecast": {},
        "risk": {},
    }

    if series.shape[0] <= holdout_periods:
        report["models"]["naive"] = {"available": False, "reason": "insufficient history"}
        report["models"]["sarima"] = {"available": False, "reason": "insufficient history"}
        report["selection"] = {"best_model": None, "reason": "insufficient history"}
        report["forecast"] = {"horizon": forecast_horizon, "values": []}
        report["risk"] = _garch_risk_summary(series.pct_change().replace([np.inf, -np.inf], np.nan).dropna())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    train = series.iloc[:-holdout_periods]
    test = series.iloc[-holdout_periods:]

    naive_test_pred = _naive_forecast(train, holdout_periods)
    naive_metrics = _evaluate(test.to_numpy(dtype=float), naive_test_pred)
    report["models"]["naive"] = {
        "available": True,
        "metrics": naive_metrics,
    }

    sarima_result = _fit_sarima_forecast(train, holdout_periods)
    if sarima_result["available"]:
        sarima_metrics = _evaluate(test.to_numpy(dtype=float), np.asarray(sarima_result["forecast"], dtype=float))
        report["models"]["sarima"] = {
            "available": True,
            "metrics": sarima_metrics,
            "aic": sarima_result["aic"],
            "order": sarima_result["order"],
            "seasonal_order": sarima_result["seasonal_order"],
        }
    else:
        report["models"]["sarima"] = {
            "available": False,
            "reason": sarima_result["reason"],
        }

    best_model = "naive"
    best_rmse = naive_metrics["rmse"]
    if sarima_result["available"]:
        sarima_rmse = report["models"]["sarima"]["metrics"]["rmse"]
        if sarima_rmse < best_rmse:
            best_model = "sarima"
            best_rmse = sarima_rmse

    report["selection"] = {
        "best_model": best_model,
        "metric": "rmse",
        "rmse": float(best_rmse),
    }

    if best_model == "sarima":
        forecast_values = np.asarray(_fit_sarima_forecast(series, forecast_horizon)["forecast"], dtype=float)
        if forecast_values.size == 0:
            forecast_values = _naive_forecast(series, forecast_horizon)
    else:
        forecast_values = _naive_forecast(series, forecast_horizon)
    report["forecast"] = {
        "horizon": int(forecast_horizon),
        "values": [float(value) for value in forecast_values],
    }

    returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    report["risk"] = _garch_risk_summary(returns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run time-series forecasting and risk analysis for a sales segment.")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--product-line", type=str, default="Classic Cars")
    parser.add_argument("--country", type=str, default="USA")
    parser.add_argument("--holdout-periods", type=int, default=6)
    parser.add_argument("--forecast-horizon", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = SalesPaths.default()
    loader = SalesDatasetLoader(random_state=42)
    frame = loader.load(csv_path=Path(args.csv_path) if args.csv_path else None, use_synthetic_if_missing=True)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else paths.reports_dir / "time_series" / "classic_cars_usa_risk.json"
    )
    report = generate_segment_time_series_report(
        frame=frame,
        output_path=output_path,
        product_line=args.product_line,
        country=args.country,
        holdout_periods=args.holdout_periods,
        forecast_horizon=args.forecast_horizon,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
