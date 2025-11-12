import argparse
import json
import re
from collections import Counter, deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests


DEFAULT_JSON_DIR = Path("data/xbrl-json")
DEFAULT_OUTPUT = Path("data/csv-data/ai_keyword_counts.csv")

# Keywords and phrases associated with AI across several languages.
KEYWORD_PATTERNS: Dict[str, re.Pattern] = {
    "artificial intelligence": re.compile(r"\bartificial intelligence\b", re.IGNORECASE),
    "ai": re.compile(r"\bai\b", re.IGNORECASE),
    "machine learning": re.compile(r"\bmachine learning\b", re.IGNORECASE),
    "generative ai": re.compile(r"\bgenerative ai\b", re.IGNORECASE),
    "intelligence artificielle": re.compile(r"\bintelligence artificielle\b", re.IGNORECASE),
    "apprentissage automatique": re.compile(r"\bapprentissage automatique\b", re.IGNORECASE),
    "inteligencia artificial": re.compile(r"\binteligencia artificial\b", re.IGNORECASE),
    "intelig\u00eancia artificial": re.compile(r"\bintelig\u00eancia artificial\b", re.IGNORECASE),
    "intelligenza artificiale": re.compile(r"\bintelligenza artificiale\b", re.IGNORECASE),
    "k\u00fcnstliche intelligenz": re.compile(r"\bk\u00fcnstliche intelligenz\b", re.IGNORECASE),
    "teko\u00e4ly": re.compile(r"\bteko\u00e4ly\b", re.IGNORECASE),
    "tehisintellekt": re.compile(r"\btehisintellekt\b", re.IGNORECASE),
}

# Namespaces we can safely ignore when looking for company-specific URLs.
NAMESPACE_BLOCKLIST = (
    "xbrl.org",
    "ifrs.org",
    "xbrl.ifrs.org",
    "www.xbrl.org",
    "iso.org",
    "xbrl.com",
    "europa.eu",
    "w3.org",
    "eurofiling.info",
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}


class SimpleHTMLExtractor(HTMLParser):
    """Minimal HTML parser to collect text and hyperlinks."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer: List[str] = []
        self.links: List[str] = []
        self._ignore_stack: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignore_stack.append(tag)
        if tag == "a":
            for attr, value in attrs:
                if attr == "href" and value:
                    self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if self._ignore_stack and self._ignore_stack[-1] == tag:
            self._ignore_stack.pop()

    def handle_data(self, data: str) -> None:
        if not self._ignore_stack:
            self._buffer.append(data)

    @property
    def text(self) -> str:
        return " ".join(chunk.strip() for chunk in self._buffer if chunk.strip())


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_entity_identifier(data: dict) -> Optional[str]:
    facts = data.get("facts", {})
    for fact in facts.values():
        dimensions = fact.get("dimensions", {})
        entity = dimensions.get("entity")
        if entity:
            return entity
    return None


def extract_candidate_urls(data: dict) -> Set[str]:
    urls: Set[str] = set()
    doc_info = data.get("documentInfo", {})

    namespaces = doc_info.get("namespaces", {}) or {}
    for candidate in namespaces.values():
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            urls.add(candidate)

    taxonomy_entries = doc_info.get("taxonomy", []) or []
    for candidate in taxonomy_entries:
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            urls.add(candidate)

    filtered: Set[str] = set()
    for url in urls:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            continue
        if any(blocked in parsed.netloc for blocked in NAMESPACE_BLOCKLIST):
            continue
        base = f"{parsed.scheme}://{parsed.netloc}"
        filtered.add(base.rstrip("/"))

    return filtered


def fetch_url(url: str, timeout: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=(5, timeout), allow_redirects=True)
        response.raise_for_status()
        return response.text, response.url
    except Exception as exc:
        print(f"[error] fetch failed for {url}: {exc}")
        return None, None


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or ""
    canonical = f"{parsed.scheme}://{parsed.netloc}{path}"
    if canonical.endswith("/") and path not in ("", "/"):
        canonical = canonical.rstrip("/")
    return canonical


def normalize_link(base_url: str, link: str, allowed_netlocs: Set[str]) -> Optional[str]:
    if not link:
        return None
    base = base_url if base_url.endswith("/") else base_url + "/"
    joined = urljoin(base, link)
    parsed = urlparse(joined)
    if not parsed.scheme.startswith("http"):
        return None
    netloc = parsed.netloc.lower()
    if allowed_netlocs and netloc not in allowed_netlocs:
        return None
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path or ''}"
    if normalized.endswith("/") and parsed.path not in ("", "/"):
        normalized = normalized.rstrip("/")
    return normalized


def crawl_domain(domain: str, max_pages: int, timeout: int) -> Tuple[int, Counter]:
    visited: Set[str] = set()
    queue: deque[str] = deque()
    canonical_domain = canonicalize_url(domain)
    queue.append(canonical_domain)
    allowed_netlocs: Set[str] = {urlparse(canonical_domain).netloc.lower()}
    keyword_counter: Counter = Counter()
    pages_scanned = 0

    while queue and pages_scanned < max_pages:
        current_url = queue.popleft()
        canonical_current = canonicalize_url(current_url)
        if canonical_current in visited:
            continue
        visited.add(canonical_current)

        print(f"[crawl] {domain}: scanning {current_url} ({pages_scanned + 1}/{max_pages})")

        html, final_url = fetch_url(current_url, timeout=timeout)
        if html is None:
            print(f"[crawl] {domain}: failed to fetch {current_url}")
            continue
        if final_url:
            final_canonical = canonicalize_url(final_url)
            allowed_netlocs.add(urlparse(final_canonical).netloc.lower())
            if final_canonical not in visited:
                visited.add(final_canonical)
            current_url = final_canonical

        parser = SimpleHTMLExtractor()
        try:
            parser.feed(html)
            parser.close()
        except Exception:
            print(f"[crawl] {domain}: parse error on {current_url}")
            continue

        text_content = parser.text.lower()
        for keyword, pattern in KEYWORD_PATTERNS.items():
            matches = pattern.findall(text_content)
            if matches:
                keyword_counter[keyword] += len(matches)
                print(f"[crawl] {domain}: found {len(matches)} match(es) for '{keyword}' on {current_url}")

        pages_scanned += 1

        for link in parser.links:
            normalized = normalize_link(current_url, link, allowed_netlocs)
            if not normalized:
                continue
            canonical_link = canonicalize_url(normalized)
            if canonical_link in visited:
                continue
            if canonical_link not in queue and len(queue) + pages_scanned < max_pages:
                queue.append(canonical_link)

    print(
        f"[crawl] {domain}: completed with {pages_scanned} page(s) scanned, "
        f"{sum(keyword_counter.values())} total AI keyword match(es)"
    )
    return pages_scanned, keyword_counter


def build_results(json_dir: Path, max_pages: int, timeout: int) -> List[Dict[str, object]]:
    max_pages = max(max_pages, 5)
    results: List[Dict[str, object]] = []
    seen_domains: Set[str] = set()

    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = load_json(json_file)
        except Exception as exc:
            print(f"Failed to read {json_file.name}: {exc}")
            continue

        entity_id = extract_entity_identifier(data) or "unknown"
        candidate_domains = extract_candidate_urls(data)

        if not candidate_domains:
            print(f"No company URLs found in {json_file.name}")
            continue

        for domain in candidate_domains:
            if domain in seen_domains:
                print(f"[skip] {domain} already processed; skipping crawl")
                results.append(
                    {
                        "entity": entity_id,
                        "domain": domain,
                        "pages_scanned": 0,
                        "ai_keyword_total": 0,
                        "top_keywords": "",
                    }
                )
                continue

            pages_scanned, keyword_counts = crawl_domain(domain, max_pages=max_pages, timeout=timeout)
            seen_domains.add(domain)

            total_matches = sum(keyword_counts.values())
            top_keywords = ", ".join(
                f"{keyword}:{count}" for keyword, count in keyword_counts.most_common() if count > 0
            )

            results.append(
                {
                    "entity": entity_id,
                    "domain": domain,
                    "pages_scanned": pages_scanned,
                    "ai_keyword_total": total_matches,
                    "top_keywords": top_keywords,
                }
            )
            print(
                f"[result] {domain}: pages={pages_scanned}, matches={total_matches}, "
                f"top_keywords='{top_keywords or 'none'}'"
            )

    return results


def dataframe_from_results(results: List[Dict[str, object]]) -> Optional["pd.DataFrame"]:
    if not results:
        return None
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract company URLs from JSON reports, crawl their websites, and count AI-related keywords."
    )
    parser.add_argument(
        "--json-dir",
        default=str(DEFAULT_JSON_DIR),
        help="Directory containing XBRL JSON filings (default: data/xbrl-json).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="CSV path to store AI keyword counts (default: data/csv-data/ai_keyword_counts.csv).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Maximum number of pages to crawl per domain (minimum enforced: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Read timeout in seconds for each request (default: 20).",
    )

    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    effective_max_pages = max(args.max_pages, 5)
    if effective_max_pages != args.max_pages:
        print(f"Requested max pages {args.max_pages} is below minimum. Using {effective_max_pages}.")

    results = build_results(json_dir=json_dir, max_pages=effective_max_pages, timeout=args.timeout)
    df = dataframe_from_results(results)

    if df is None or df.empty:
        print("No results generated. No output file created.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(by=["ai_keyword_total", "domain"], ascending=[False, True]).to_csv(output_path, index=False)

    print(f"AI keyword counts saved to: {output_path}")
    print(f"Domains processed: {df['domain'].nunique()}")
    print(f"Rows written: {len(df)}")


if __name__ == "__main__":
    import pandas as pd

    main()

