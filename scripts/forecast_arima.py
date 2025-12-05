import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_INPUT = Path("data/csv-data/financial_ratios.csv")
DEFAULT_OUTPUT_DIR = Path("data/forecast-results")


def load_ratio_series(
    path: Path,
    entity: str | None,
    company_name: str | None,
    metric: str,
) -> pd.Series:
    df = pd.read_csv(path)

    if entity:
        df = df[df["entity"] == entity]
    if company_name:
        df = df[df["company_name"] == company_name]

    if df.empty:
        raise ValueError("No rows matched the given entity / company_name filters.")

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in columns: {list(df.columns)}")

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df = df.sort_values("year")

    if df.empty:
        raise ValueError("No valid year values available after cleaning.")

    s = df.set_index("year")[metric].astype(float)
    s = s.groupby(level=0).mean()
    return s


def fit_arima(series: pd.Series):
    if len(series) <= 2:
        raise ValueError("Need at least 3 observations to fit ARIMA.")

    if len(series) <= 3:
        order = (0, 1, 0)
    else:
        order = (1, 1, 1)

    model = sm.tsa.ARIMA(series, order=order)
    return model.fit()


def forecast_arima(
    series: pd.Series,
    horizon: int,
) -> tuple[pd.Series, pd.DataFrame]:
    model_res = fit_arima(series)
    forecast_res = model_res.get_forecast(steps=horizon)
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    last_year = int(series.index.max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon, dtype=int)
    mean_forecast.index = future_years
    conf_int.index = future_years

    return mean_forecast, conf_int


def plot_forecast(
    series: pd.Series,
    forecast: pd.Series,
    conf_int: pd.DataFrame,
    metric: str,
    title_suffix: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(series.index, series.values, label="Historical", marker="o")

    plt.plot(forecast.index, forecast.values, label="Forecast", marker="o", linestyle="--")

    plt.fill_between(
        forecast.index,
        conf_int.iloc[:, 0].values,
        conf_int.iloc[:, 1].values,
        color="gray",
        alpha=0.3,
        label="95% CI",
    )

    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.title(f"ARIMA Forecast for {metric} {title_suffix}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARIMA forecasting for financial ratios by entity / company."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to financial_ratios CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help=(
            "Ratio column to forecast, e.g.: profit_margin, return_on_assets, "
            "return_on_equity, assets_turnover, cash_flow_to_assets, cash_flow_to_equity"
        ),
    )
    parser.add_argument(
        "--entity",
        help="Filter by LEI / entity code (matches the 'entity' column).",
    )
    parser.add_argument(
        "--company-name",
        help="Filter by company name (matches the 'company_name' column).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Number of future years to forecast (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save forecast CSVs and plots (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    series = load_ratio_series(
        input_path,
        entity=args.entity,
        company_name=args.company_name,
        metric=args.metric,
    )

    if len(series) < 3:
        raise ValueError(
            f"Not enough data points for ARIMA. Need at least 3 years, found {len(series)}."
        )

    forecast, conf_int = forecast_arima(series, args.horizon)

    hist_df = series.reset_index()
    hist_df.columns = ["year", "historical_value"]

    fc_df = forecast.reset_index()
    fc_df.columns = ["year", "forecast_value"]
    ci_df = conf_int.reset_index()
    ci_df.columns = ["year", "lower_95", "upper_95"]

    result_df = pd.merge(hist_df, fc_df, on="year", how="outer")
    result_df = pd.merge(result_df, ci_df, on="year", how="left")

    safe_metric = args.metric.replace("/", "_")
    suffix_parts = []
    if args.entity:
        suffix_parts.append(args.entity)
    if args.company_name:
        suffix_parts.append(args.company_name.replace(" ", "_"))
    suffix = "_".join(suffix_parts) if suffix_parts else "all_entities"

    csv_path = output_dir / f"arima_{safe_metric}_{suffix}.csv"
    png_path = output_dir / f"arima_{safe_metric}_{suffix}.png"

    result_df.to_csv(csv_path, index=False)

    title_suffix = f"({suffix})"
    plot_forecast(series, forecast, conf_int, args.metric, title_suffix, png_path)

    print(f"ARIMA forecast saved to: {csv_path}")
    print(f"Forecast plot saved to: {png_path}")


if __name__ == "__main__":
    main()


