#!/usr/bin/env python3
"""
Standalone downloader for filings.xbrl.org API (JSON:API).
Docs: https://filings.xbrl.org/docs/api
"""
import argparse
from urllib.parse import urlencode
from pathlib import Path
import requests

API_BASE = "https://filings.xbrl.org/api/filings"

def api_get(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def iter_filings(query: str, limit: int):
    url = f"{API_BASE}?{query}" if query else API_BASE
    count = 0
    while url:
        data = api_get(url)
        for item in data.get("data", []):
            yield item
            count += 1
            if limit and count >= limit:
                return
        url = (data.get("links") or {}).get("next")

def first_attr(attrs: dict, *keys, default=None):
    for k in keys:
        if k in attrs and attrs[k]:
            return attrs[k]
    return default

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--country')
    p.add_argument('--filter')
    p.add_argument('--include', default='entity')
    p.add_argument('--sort', default='-processed')
    p.add_argument('--page-size', type=int, default=200)
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--out-json', default='data/samples/xbrl-json')
    p.add_argument('--out-packages', default='out/packages')
    p.add_argument('--skip-json', action='store_true')
    p.add_argument('--skip-zip', action='store_true')
    args = p.parse_args()

    qp = {}
    if args.country:
        qp['filter[country]'] = args.country
    if args.include:
        qp['include'] = args.include
    if args.sort:
        qp['sort'] = args.sort
    qp['page[size]'] = min(args.page_size or 200, 200)
    query = args.filter if args.filter else urlencode(qp)

    out_json = Path(args.out_json); out_json.mkdir(parents=True, exist_ok=True)
    out_zip = Path(args.out_packages); out_zip.mkdir(parents=True, exist_ok=True)

    n = 0
    for filing in iter_filings(query, limit=args.limit):
        attrs = filing.get('attributes', {})
        filing_id = filing.get('id') or attrs.get('filing_index') or f'filing_{n+1}'
        json_url    = first_attr(attrs, 'json_url', 'xbrl_json_url')
        package_url = first_attr(attrs, 'package_url', 'report_package_url')

        if json_url and not args.skip_json:
            pth = out_json / f"{filing_id}.json"
            try:
                r = requests.get(json_url, timeout=60)
                r.raise_for_status()
                pth.write_bytes(r.content)
                print('[JSON]', pth)
            except Exception as e:
                print('JSON failed:', filing_id, e)

        if package_url and not args.skip_zip:
            pthz = out_zip / f"{filing_id}.zip"
            try:
                r = requests.get(package_url, timeout=120)
                r.raise_for_status()
                pthz.write_bytes(r.content)
                print('[ZIP ]', pthz)
            except Exception as e:
                print('ZIP failed:', filing_id, e)
        n += 1

    print('Done, files:', n)

if __name__ == '__main__':
    main()
