"""
Scraper for the "Asanas & Mudras" article category on habuild.in/habitology.

The habitology section is served by a WordPress backend at wp.habuild.in.
Rather than scraping the fragile, JS-rendered listing page directly, this
script talks to WordPress's public REST API (the same data source the
website itself uses), which returns clean, paginated, structured data for
every article in the category -- including the full article body -- in a
handful of requests.

Usage:
    python habitology_scraper.py
    python habitology_scraper.py --output-dir output --limit 20
    python habitology_scraper.py --delay 1.0 --per-page 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

WP_API_BASE = "https://wp.habuild.in/wp-json/wp/v2"
CATEGORY_SLUG = "asana-mudra"
USER_AGENT = "habitology-asana-mudra-scraper/1.0 (+https://habuild.in/habitology/asana-mudra/)"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3


@dataclass
class Article:
    id: int
    title: str
    url: str
    published: str
    modified: str
    excerpt: str
    content: str
    tags: list[str]


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def request_with_retries(session: requests.Session, url: str, params: dict) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code} for {response.url}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts: {last_error}")


def get_category_id(session: requests.Session, slug: str) -> int:
    response = request_with_retries(session, f"{WP_API_BASE}/categories", {"slug": slug})
    results = response.json()
    if not results:
        raise RuntimeError(f"No category found with slug '{slug}'")
    return results[0]["id"]


def get_tag_names(session: requests.Session, tag_ids: list[int], tag_cache: dict[int, str]) -> list[str]:
    names = []
    for tag_id in tag_ids:
        if tag_id not in tag_cache:
            response = request_with_retries(session, f"{WP_API_BASE}/tags/{tag_id}", {})
            tag_cache[tag_id] = response.json().get("name", "")
        if tag_cache[tag_id]:
            names.append(tag_cache[tag_id])
    return names


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def public_url(link: str) -> str:
    """The API reports links on the wp.habuild.in backend; habuild.in serves the same
    pages publicly, which is the domain the user actually browses."""
    return link.replace("https://wp.habuild.in/", "https://habuild.in/", 1)


def slugify(value: str, max_len: int = 80) -> str:
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[\s_-]+", "-", value)
    return value[:max_len].strip("-") or "untitled"


def iter_category_posts(
    session: requests.Session,
    category_id: int,
    per_page: int,
    delay: float,
    limit: int | None,
) -> Iterator[dict]:
    page = 1
    fetched = 0
    while True:
        response = request_with_retries(
            session,
            f"{WP_API_BASE}/posts",
            {"categories": category_id, "per_page": per_page, "page": page, "orderby": "date", "order": "desc"},
        )
        posts = response.json()
        if not posts:
            return
        for post in posts:
            yield post
            fetched += 1
            if limit is not None and fetched >= limit:
                return
        total_pages = int(response.headers.get("X-WP-TotalPages", page))
        if page >= total_pages:
            return
        page += 1
        time.sleep(delay)


def scrape(output_dir: Path, per_page: int, delay: float, limit: int | None) -> list[Article]:
    session = make_session()

    print(f"Looking up category id for '{CATEGORY_SLUG}'...")
    category_id = get_category_id(session, CATEGORY_SLUG)
    print(f"Category '{CATEGORY_SLUG}' -> id {category_id}")

    tag_cache: dict[int, str] = {}
    articles: list[Article] = []

    for post in iter_category_posts(session, category_id, per_page, delay, limit):
        tag_names = get_tag_names(session, post.get("tags", []), tag_cache)
        article = Article(
            id=post["id"],
            title=clean_html(post["title"]["rendered"]),
            url=public_url(post["link"]),
            published=post["date"],
            modified=post["modified"],
            excerpt=clean_html(post["excerpt"]["rendered"]),
            content=clean_html(post["content"]["rendered"]),
            tags=tag_names,
        )
        articles.append(article)
        print(f"[{len(articles)}] {article.title}")
        time.sleep(delay)

    save_json(articles, output_dir / "asana_mudra_articles.json")
    save_csv(articles, output_dir / "asana_mudra_index.csv")
    save_text_files(articles, output_dir / "articles")

    return articles


def save_json(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(articles)} articles to {path}")


def save_csv(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "url", "published", "modified", "tags"])
        for a in articles:
            writer.writerow([a.id, a.title, a.url, a.published, a.modified, "; ".join(a.tags)])
    print(f"Wrote index CSV to {path}")


def save_text_files(articles: list[Article], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for a in articles:
        filename = f"{a.id}-{slugify(a.title)}.txt"
        body = (
            f"{a.title}\n"
            f"{'=' * len(a.title)}\n\n"
            f"URL: {a.url}\n"
            f"Published: {a.published}\n"
            f"Tags: {', '.join(a.tags) if a.tags else '-'}\n\n"
            f"{a.content}\n"
        )
        (directory / filename).write_text(body, encoding="utf-8")
    print(f"Wrote {len(articles)} article text files to {directory}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory to write results into")
    parser.add_argument("--per-page", type=int, default=100, help="Posts per API page (max 100)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay in seconds between requests")
    parser.add_argument("--limit", type=int, default=None, help="Only scrape the first N articles (for testing)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    articles = scrape(args.output_dir, args.per_page, args.delay, args.limit)
    print(f"\nDone. Scraped {len(articles)} Asanas & Mudras articles.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
