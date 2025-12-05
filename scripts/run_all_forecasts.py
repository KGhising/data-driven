import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from plot_all_forecasts import (
    plot_all_metrics_single_window,
)

RATIOS_CSV = Path("data/csv-data/financial_ratios.csv")
FORECAST_DIR = Path("data/forecast-results")


def build_series(df: pd.DataFrame, entity: str, metric: str) -> pd.Series | None:
    subset = df[df["entity"] == entity].copy()
    if subset.empty:
        return None

    subset["year"] = pd.to_numeric(subset["year"], errors="coerce")
    subset = subset.dropna(subset=["year"])
    subset = subset.sort_values("year")
    if subset.empty:
        return None

    s = subset.set_index("year")[metric]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 3:
        return None

    s = s.groupby(level=0).mean()
    return s


def fit_and_forecast(series: pd.Series, horizon: int) -> tuple[pd.Series, pd.DataFrame]:
    if len(series) <= 3:
        order = (0, 1, 0)
    else:
        order = (1, 1, 1)

    model = sm.tsa.ARIMA(series, order=order)
    res = model.fit()

    fc = res.get_forecast(steps=horizon)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    last_year = int(series.index.max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon, dtype=int)
    mean_fc.index = future_years
    ci.index = future_years

    return mean_fc, ci


def save_forecast_csv(
    series: pd.Series,
    forecast: pd.Series,
    conf_int: pd.DataFrame,
    metric: str,
    entity: str,
    company_name: str,
    output_dir: Path,
) -> Path:
    hist_df = series.reset_index()
    hist_df.columns = ["year", "historical_value"]

    fc_df = forecast.reset_index()
    fc_df.columns = ["year", "forecast_value"]
    ci_df = conf_int.reset_index()
    ci_df.columns = ["year", "lower_95", "upper_95"]

    result_df = pd.merge(hist_df, fc_df, on="year", how="outer")
    result_df = pd.merge(result_df, ci_df, on="year", how="left")

    safe_metric = metric.replace("/", "_")
    safe_company = company_name.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"arima_{safe_metric}_{safe_company}.csv"

    result_df.to_csv(csv_path, index=False)
    return csv_path


def run_all(horizon: int) -> None:
    if not RATIOS_CSV.exists():
        raise FileNotFoundError(f"Ratios CSV not found: {RATIOS_CSV}")

    df = pd.read_csv(RATIOS_CSV)

    metric_cols = [
        c
        for c in df.columns
        if c not in {"entity", "company_name", "year"}
    ]
    entity_company_pairs = (
        df[["entity", "company_name"]]
        .dropna(subset=["entity", "company_name"])
        .drop_duplicates()
        .to_records(index=False)
    )

    print(f"Found {len(entity_company_pairs)} entities and {len(metric_cols)} metrics")

    records: list[dict] = []

    for metric in metric_cols:
        for entity, company_name in entity_company_pairs:
            series = build_series(df, entity, metric)
            if series is None or len(series) < 3:
                continue
            try:
                forecast, ci = fit_and_forecast(series, horizon=horizon)
            except Exception as exc:
                print(
                    f"[warn] Failed ARIMA for entity={entity}, "
                    f"company_name={company_name}, metric={metric}: {exc}"
                )
                continue
            csv_path = save_forecast_csv(
                series,
                forecast,
                ci,
                metric,
                entity,
                company_name,
                FORECAST_DIR,
            )
            print(f"Saved forecast: {csv_path}")

            for year, val in series.items():
                records.append(
                    {
                        "metric": metric,
                        "label": company_name,
                        "year": float(year),
                        "value": float(val),
                        "is_forecast": False,
                    }
                )

            for year, val in forecast.items():
                records.append(
                    {
                        "metric": metric,
                        "label": company_name,
                        "year": float(year),
                        "value": float(val),
                        "is_forecast": True,
                    }
                )

    if not records:
        print("No forecasts produced; nothing to plot.")
        return

    all_df = pd.DataFrame(records)

    json_path = FORECAST_DIR / "all_forecasts.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_json(json_path, orient="records")
    print(f"Wrote combined JSON for web viewer to: {json_path}")

    plot_all_metrics_single_window(all_df, FORECAST_DIR, interactive=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ARIMA forecasts for ALL entities and ALL metrics at once "
            "and display a single interactive window with all ratios."
        )
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Number of future years to forecast for each series (default: 3).",
    )

    args = parser.parse_args()
    run_all(horizon=args.horizon)


if __name__ == "__main__":
    main()


