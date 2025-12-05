import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_INPUT_DIR = Path("data/forecast-results")
DEFAULT_OUTPUT_DIR = Path("data/forecast-results")


def parse_forecast_filename(path: Path) -> Tuple[str, str]:
    stem = path.stem
    parts = stem.split("_", 2)
    if len(parts) < 3 or parts[0] != "arima":
        raise ValueError(f"Unrecognized forecast filename pattern: {path.name}")
    metric = parts[1]
    label_raw = parts[2]
    label = label_raw.replace("_", " ")
    return metric, label


def load_all_forecasts(
    input_dir: Path,
    metric_filter: str | None = None,
) -> pd.DataFrame:
    files: List[Path] = sorted(input_dir.glob("arima_*.csv"))
    if not files:
        raise FileNotFoundError(f"No forecast CSV files found in {input_dir}")

    records = []
    for path in files:
        metric, label = parse_forecast_filename(path)
        if metric_filter and metric != metric_filter:
            continue

        df = pd.read_csv(path)
        if "year" not in df.columns:
            continue

        df = df.copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"])

        hist = pd.to_numeric(df.get("historical_value"), errors="coerce")
        fc = pd.to_numeric(df.get("forecast_value"), errors="coerce")

        value = hist.where(hist.notna(), fc)
        is_forecast = hist.isna() & fc.notna()

        for y, v, f in zip(df["year"], value, is_forecast):
            if pd.isna(v):
                continue
            records.append(
                {
                    "metric": metric,
                    "label": label,
                    "year": float(y),
                    "value": float(v),
                    "is_forecast": bool(f),
                }
            )

    if not records:
        raise ValueError("No usable forecast data found.")

    return pd.DataFrame(records)


def plot_forecasts_for_metric(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    interactive: bool = False,
) -> Path | None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    subset = df[df["metric"] == metric].copy()
    if subset.empty:
        raise ValueError(f"No data for metric '{metric}'.")

    g = subset.copy()
        g["kind"] = g["is_forecast"].map({False: "historical", True: "forecast"})
        sns.lineplot(
            ax=plt.gca(),
            data=g,
            x="year",
            y="value",
            hue="label",
            style="kind",
            markers=True,
            dashes={"historical": "", "forecast": (2, 2)},
            linewidth=1.8,
        )

    plt.xlabel("Year")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"ARIMA Forecasts for {metric.replace('_', ' ').title()}")
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = output_dir / f"all_forecasts_{metric}.png"

    if interactive:
        plt.show()
        return None
    else:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path


def plot_all_metrics_single_window(
    df: pd.DataFrame,
    output_dir: Path,
    interactive: bool = False,
) -> Path | None:
    sns.set(style="whitegrid")

    metrics = sorted(df["metric"].unique())
    n = len(metrics)
    if n == 0:
        raise ValueError("No metrics to plot.")

    ncols = int(round(n ** 0.5))
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        subset = df[df["metric"] == metric].copy()
        if subset.empty:
            continue

        g = subset.copy()
        g["kind"] = g["is_forecast"].map({False: "historical", True: "forecast"})
        sns.lineplot(
            ax=ax,
            data=g,
            x="year",
            y="value",
            hue="label",
            style="kind",
            markers=True,
            dashes={"historical": "", "forecast": (2, 2)},
            linewidth=1.5,
        )

        ax.set_xlabel("Year")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(alpha=0.3)
        ax.legend(
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            ncol=2,
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("ARIMA Forecasts for All Financial Ratios", fontsize=14)
    plt.tight_layout(rect=(0, 0, 0.85, 0.97))

    output_path = output_dir / "all_forecasts_all_metrics.png"

    if interactive:
        plt.show()
        return None
    else:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all ARIMA forecasts together using matplotlib/seaborn."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing arima_*.csv forecast files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save combined forecast plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--metric",
        help=(
            "If provided, only plot this metric (e.g. profit_margin). "
            "If omitted, plots one figure per metric found."
        ),
    )
    parser.add_argument(
        "--single-window",
        action="store_true",
        help=(
            "Plot all metrics and all entities in a single window "
            "(one subplot per metric)."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show interactive matplotlib windows instead of saving PNG files.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_forecasts(input_dir, metric_filter=args.metric)

    if args.single_window and not args.metric:
        path = plot_all_metrics_single_window(df, output_dir, interactive=args.interactive)
        if not args.interactive and path is not None:
            print(f"Saved single-window plot for all metrics to: {path}")
    else:
        metrics = [args.metric] if args.metric else sorted(df["metric"].unique())
        for metric in metrics:
            path = plot_forecasts_for_metric(df, metric, output_dir, interactive=args.interactive)
            if not args.interactive and path is not None:
                print(f"Saved combined forecast plot for {metric} to: {path}")


if __name__ == "__main__":
    main()


