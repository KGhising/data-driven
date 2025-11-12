# CSRD-ESRS XBRL Data Tools

Tools for downloading and analyzing CSRD/ESRS XBRL filings from the filings.xbrl.org API with support for ESRS / ESEF Taxonomy fact extraction.

## Features

- Download XBRL filings from the filings.xbrl.org API
- Export all facts directly to tabular format (pivot table)
- Filter and export ESRS / ESEF Taxonomy-specific facts to tabular format
- Extract taxonomy information (schemas, namespaces, versions)
- Tabular reports with concepts as rows and periods as columns
- Support for batch processing multiple files or single file processing

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

### 1. Clone the repository

```bash
git clone "https://github.com/KGhising/csrd-esrs.git"
cd csrd-esrs
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
pip install pandas matplotlib jupyter requests
```

**Optional visualization packages:**
```bash
pip install seaborn plotly
```

### 4. Create data directories

```bash
mkdir -p data/xbrl-json data/csv-data
```

## Usage

### Download XBRL Filings

Download filings from the filings.xbrl.org API:

```bash
# Download 5 filings from Estonia
python scripts/fetch_filings.py --country EE --limit 5

# Download from a specific country
python scripts/fetch_filings.py --country FR --limit 10

# Download all default filings (up to 50)
python scripts/fetch_filings.py

# Download only filings processed after 2024
python scripts/fetch_filings.py --from-year 2024

# Download filings after a specific date
python scripts/fetch_filings.py --after-date 2024-01-01

# Download filings from a specific company
python scripts/fetch_filings.py --company "Apple" --limit 10

# Combine filters: country, date, and company
python scripts/fetch_filings.py --country EE --from-year 2024 --company "Silvano" --limit 20
```

**Arguments:**
- `--country`: Filter by country code (e.g., EE, FR, DE)
- `--company`: Filter by company/entity name (case-insensitive partial match, searches in entity name or identifier)
- `--from-year`: Filter filings processed from this year onwards (e.g., `2024` for all filings after 2023)
- `--after-date`: Filter filings processed after this date (format: `YYYY-MM-DD` or just `YYYY`)
- `--limit`: Maximum number of filings to download (default: 50)
- `--out-json`: Output directory (default: `data/xbrl-json`)

### Export Facts to Tabular Format

Both export scripts create tabular reports directly (pivot tables with concepts as rows and periods as columns).

#### Export All Facts

```bash
# Export all facts from all JSON files to tabular format
python scripts/export_facts.py

# Export with custom output path
python scripts/export_facts.py --output data/csv-data/my_facts_tabular.csv

# Process a single file
python scripts/export_facts.py --file data/xbrl-json/15608.json

# Custom input directory
python scripts/export_facts.py --input-dir data/xbrl-json/custom
```

#### Export ESRS / ESEF Taxonomy Facts Only

```bash
# Export only ESRS / ESEF Taxonomy facts from all files to tabular format
python scripts/export_eu_taxonomy_facts.py

# Process a single file
python scripts/export_eu_taxonomy_facts.py --file data/xbrl-json/15608.json

# Custom output path
python scripts/export_eu_taxonomy_facts.py --output data/csv-data/eu_taxonomy_tabular.csv

# Custom input directory
python scripts/export_eu_taxonomy_facts.py --input-dir data/xbrl-json/custom
```

**Arguments for export scripts:**
- `--input-dir`: Directory containing XBRL JSON files (default: `data/xbrl-json`)
- `--output`: Output CSV file path for tabular report
- `--file`: Process a single file instead of directory

### Extract Taxonomy Information

Extract taxonomy information from concept dimensions in facts. Extracts taxonomies (namespace prefixes like "ifrs-full") and their clean concept names (e.g., "AdministrativeExpense" from "ifrs-full:AdministrativeExpense"):

```bash
# Extract taxonomy from all JSON files
python scripts/extract_taxonomy.py

# Process a single file
python scripts/extract_taxonomy.py --file data/xbrl-json/15608.json

# Custom output path
python scripts/extract_taxonomy.py --output data/csv-data/my_taxonomy_concepts.csv
```

**Arguments:**
- `--input-dir`: Directory containing XBRL JSON files (default: `data/xbrl-json`)
- `--output`: Output CSV file path for taxonomy-concept details (default: `data/csv-data/taxonomy_concepts.csv`)
- `--file`: Process a single file instead of directory

The script generates one CSV file:
- **Taxonomy Concepts** (`taxonomy_concepts.csv`): Unique taxonomy-concept pairs aggregated across all files (taxonomy and concept columns only)

## Script Overview

| Script | Purpose | Default Output |
|--------|---------|----------------|
| `fetch_filings.py` | Download XBRL JSON files from API | `data/xbrl-json/` |
| `export_facts.py` | Export all facts to tabular format | `data/csv-data/facts_tabular.csv` |
| `export_eu_taxonomy_facts.py` | Export ESRS / ESEF Taxonomy facts to tabular format | `data/csv-data/eu_taxonomy_facts_tabular.csv` |
| `extract_taxonomy.py` | Extract taxonomy and concepts from fact dimensions | `data/csv-data/taxonomy_concepts.csv` |

## CSV Output Format

### Tabular Report Format

All export scripts generate tabular reports in CSV format with a pivot table structure:
- **Rows**: Company and financial concepts (data separated by company)
- **Columns**: Years extracted from reporting periods (2023, 2024, etc.)
- **Values**: Fact values for each company-concept-year combination
- **HTML cleaning**: Values are automatically cleaned of HTML tags and entities
- **Company identification**: Uses entity identifier or source filename to identify companies

The tabular format is ideal for:
- Comparing values across periods and companies in spreadsheet applications
- Quick analysis and visualization
- Creating financial reports and dashboards
- Multi-company analysis and benchmarking

**Example output structure:**
```
company,display_concept,2023,2024
CompanyA,CashAndCashEquivalents,1000000,1200000
CompanyA,TotalAssets,5000000,5500000
CompanyB,CashAndCashEquivalents,500000,600000
CompanyB,TotalAssets,3000000,3200000
...
```

### Taxonomy Information Format

The taxonomy extraction script generates one CSV file:

**Taxonomy Concepts** (`taxonomy_concepts.csv`):
- `taxonomy`: Taxonomy namespace prefix (e.g., `ifrs-full`, `iso4217`)
- `concept`: Clean concept name without taxonomy prefix (e.g., `AdministrativeExpense` from `ifrs-full:AdministrativeExpense`)

**Note**: Only unique taxonomy-concept pairs are exported, aggregated across all files. Each `(taxonomy, concept)` combination appears only once.

## Project Structure

```
csrd-esrs/
├── data/
│   ├── xbrl-json/          # Downloaded XBRL JSON files
│   └── csv-data/           # Exported tabular CSV files
├── scripts/
│   ├── fetch_filings.py              # Download filings from API
│   ├── export_facts.py               # Export all facts to tabular format
│   ├── export_eu_taxonomy_facts.py   # Export ESRS / ESEF Taxonomy facts to tabular format
│   └── extract_taxonomy.py          # Extract taxonomy information
├── .gitignore
└── Readme.md
```

## API Reference

The tools interact with the filings.xbrl.org API:
- **Base URL**: https://filings.xbrl.org/api/filings
- **Documentation**: https://filings.xbrl.org/docs/api

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
