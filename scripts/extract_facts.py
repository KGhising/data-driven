import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_INPUT_DIR = Path("data/xbrl-json")
DEFAULT_OUTPUT = Path("data/csv-data/financial_facts.csv")

TARGET_CONCEPTS = [
    "TotalEquity",
    "EquityAndLiabilities",
    "Assets",
    "Revenue",
    "ProfitLoss",
    "ProfitLossFromOperatingActivitiesBeforeInterestTaxesDepreciationAndAmortisationExpense",
    "CashFlowsFromUsedInOperatingActivities",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_period(period: Optional[str]) -> str:
    if not period:
        return "Unknown"
    text = str(period)
    if "T" in text and "/" in text:
        text = text.split("/", 1)[0]
    if "T" in text:
        return text.split("T")[0][:4]
    if "/" in text:
        return text.split("/", 1)[0][:4]
    return text[:4]


def extract_row(fact_id: str, fact: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
    dimensions = fact.get("dimensions", {})
    concept: str = dimensions.get("concept", "")
    if not concept:
        return None

    if ":" in concept:
        namespace, name = concept.split(":", 1)
    else:
        namespace, name = "", concept

    if name not in TARGET_CONCEPTS:
        return None

    value = fact.get("value")
    if value is None:
        return None

    entity = dimensions.get("entity", "Unknown")
    period = dimensions.get("period")

    return {
        "fact_id": fact_id,
        "concept": name,
        "concept_namespace": namespace,
        "entity": entity,
        "period": period,
        "value": value,
        "source_file": source_file,
    }


def gather_concepts(files: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in files:
        try:
            data = load_json(path)
        except Exception as exc:
            print(f"[warn] Failed to read {path.name}: {exc}")
            continue

        facts = data.get("facts", {})
        for fact_id, fact in facts.items():
            record = extract_row(fact_id, fact, path.name)
            if record:
                rows.append(record)

    return rows


def to_tabular(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["year"] = df["period"].apply(clean_period)
    df["company"] = df["entity"].apply(lambda x: str(x).split("/")[-1] if "/" in str(x) else str(x))

    pivot = df.pivot_table(
        index=["company", "concept"],
        columns="year",
        values="value",
        aggfunc="first",
    )
    pivot = pivot.reset_index()
    pivot = pivot.sort_values(["company", "concept"]).reset_index(drop=True)
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export key concepts directly from XBRL JSON filings."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing XBRL JSON files (default: data/xbrl-json).",
    )
    parser.add_argument(
        "--file",
        help="Process a single JSON file instead of the entire directory.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the tabular CSV (default: data/csv-data/key_concepts_tabular_from_json.csv).",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(input_dir.glob("*.json"))

    if not files:
        print(f"No JSON files found in {input_dir if not args.file else Path(args.file).parent}")
        return

    rows = gather_concepts(files)
    if not rows:
        print("No key concept facts found in the provided files.")
        return

    tabular = to_tabular(rows)
    if tabular.empty:
        print("No tabular data produced.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabular.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Key concept tabular data saved to: {output_path}")
    print(f"Shape: {tabular.shape[0]} rows × {tabular.shape[1]} columns")


if __name__ == "__main__":
    main()

