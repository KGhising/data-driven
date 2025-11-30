## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

### 1. Clone the repository

```bash
git clone "https://github.com/KGhising/data-driven.git"
cd data-driven
```

### 2. Set up a virtual environment

**On macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install pandas requests tqdm
```

**Optional visualization packages:**
```bash
pip install matplotlib jupyter seaborn plotly
```

### 4. Create data directories

```bash
mkdir -p data/xbrl-packages data/ixbrl-reports data/csv-data
```

## Usage

### Step 1: Download XBRL Packages or iXBRL Reports

#### Option A: Download XBRL Package Files

Download ZIP package and JSON files from the filings.xbrl.org API:

```bash
# Package ZIP
python scripts/download_packages.py
# JSON file
python scripts/fetch_filings.py
# Filters: country, date, and company
python scripts/download_packages.py --country EE --from-year 2024 --company "Silvano" --limit 20
```

### Step 2: Extract iXBRL Reports from Packages (if using packages)

```bash
python scripts/extract_ixbrl_reports.py

```

### Step 3: Extract Financial Facts from iXBRL Reports

```bash
# Extract facts from all iXBRL reports
python scripts/extract_facts.py
```

### Step 4: Clean Financial Facts (Optional)

```bash
python scripts/clean_financial_facts.py
# Clean for specific years AND concepts
python scripts/clean_financial_facts.py --years 2023 2024 --concepts Equity Revenue
```
### Step 5: Calculate ratios

```bash
python scripts/calculate_ratios.py
```
### Step 6: Run forecast models

```bash
python scripts/forecast_arima.py
python scripts/plot_all_forecasts.py
python scripts/run_all_forecasts.py
```
### Step 6: View on local host

```bash
# Run server
python -m http.server
# Open on browser
http://localhost:8000/web/forecast_viewer.html
```

## API Reference

The tools interact with the filings.xbrl.org API:
- **Base URL**: https://filings.xbrl.org/api/filings
- **Documentation**: https://filings.xbrl.org/docs/api

## License

This project is open source and available under the MIT License.
