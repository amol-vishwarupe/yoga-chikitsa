# Habitology Asana & Mudra Scraper

Scrapes every article in the **Asanas & Mudras** category of
https://habuild.in/habitology/ (586 articles at the time of writing).

## How it works

The habitology section is a WordPress site (backend at `wp.habuild.in`).
Instead of parsing the JS-rendered listing page, this script calls
WordPress's public REST API directly to fetch each article's title, URL,
publish date, excerpt, and full body content — the same data the site
itself renders, but structured and reliably paginated.

## Setup

```powershell
cd scraper
python -m venv venv          # already created; recreate if needed
.\venv\Scripts\pip install -r requirements.txt
```

## Usage

```powershell
.\venv\Scripts\python habitology_scraper.py
```

Options:

- `--output-dir PATH` — where results go (default `output`)
- `--limit N` — only scrape the first N articles (useful for testing)
- `--per-page N` — API page size, max 100 (default 100)
- `--delay SECONDS` — pause between requests, be polite to the server (default 0.3)

Example (quick test run):

```powershell
.\venv\Scripts\python habitology_scraper.py --limit 10 --output-dir output_sample
```

## Output

Inside the output directory:

- `asana_mudra_articles.json` — full structured data for every article (id, title, url, dates, excerpt, content, tags)
- `asana_mudra_index.csv` — spreadsheet-friendly index
- `articles/<id>-<slug>.txt` — one plain-text file per article with the full body
