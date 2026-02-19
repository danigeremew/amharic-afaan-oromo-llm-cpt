from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import robotparser
from urllib.parse import parse_qsl, urlencode, urljoin, urldefrag, urlparse, urlunparse

import requests
import yaml
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

try:
    import langid  # type: ignore
except Exception:  # pragma: no cover
    langid = None


DetectorFactory.seed = 0

ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[።.!?፧])\s+")
TRACKING_QUERY_KEYS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}


@dataclass
class ScraperConfig:
    output_jsonl: Path
    state_db: Path
    seeds_file: Path
    user_agent: str
    request_timeout_seconds: int
    request_retries: int
    request_backoff_seconds: float
    verify_ssl: bool
    respect_robots_txt: bool
    min_sentence_chars: int
    min_ethiopic_ratio: float
    max_depth: int
    max_pages_per_cycle: int
    max_queue_size: int
    crawl_delay_seconds: float
    sleep_between_cycles_seconds: float
    revisit_after_seconds: int
    same_domain_only: bool
    allow_subdomains: bool
    allowed_domains: list[str]


def load_config(config_path: Path) -> ScraperConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return ScraperConfig(
        output_jsonl=Path(raw.get("output_jsonl", "data_gathering/am.jsonl")),
        state_db=Path(raw.get("state_db", "data_gathering/state/scraper_state.db")),
        seeds_file=Path(raw.get("seeds_file", "data_gathering/seeds.txt")),
        user_agent=str(raw.get("user_agent", "AmharicDataGatherer/1.0")),
        request_timeout_seconds=int(raw.get("request_timeout_seconds", 15)),
        request_retries=int(raw.get("request_retries", 2)),
        request_backoff_seconds=float(raw.get("request_backoff_seconds", 1.0)),
        verify_ssl=bool(raw.get("verify_ssl", True)),
        respect_robots_txt=bool(raw.get("respect_robots_txt", True)),
        min_sentence_chars=int(raw.get("min_sentence_chars", 20)),
        min_ethiopic_ratio=float(raw.get("min_ethiopic_ratio", 0.40)),
        max_depth=int(raw.get("max_depth", 2)),
        max_pages_per_cycle=int(raw.get("max_pages_per_cycle", 200)),
        max_queue_size=int(raw.get("max_queue_size", 3000)),
        crawl_delay_seconds=float(raw.get("crawl_delay_seconds", 1.0)),
        sleep_between_cycles_seconds=float(raw.get("sleep_between_cycles_seconds", 20)),
        revisit_after_seconds=int(raw.get("revisit_after_seconds", 86400)),
        same_domain_only=bool(raw.get("same_domain_only", True)),
        allow_subdomains=bool(raw.get("allow_subdomains", True)),
        allowed_domains=[str(x).strip().lower() for x in raw.get("allowed_domains", []) if str(x).strip()],
    )


def read_seeds(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Seeds file not found: {path}")

    seeds: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            seeds.append(value)
    if not seeds:
        raise ValueError(f"No valid seeds in: {path}")
    return seeds


def canonicalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""

    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    clean_qs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        if key.lower() in TRACKING_QUERY_KEYS:
            continue
        clean_qs.append((key, value))
    clean_qs.sort()

    query = urlencode(clean_qs, doseq=True)
    canonical = urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))
    canonical, _ = urldefrag(canonical)
    return canonical


def domain_from_url(url: str) -> str:
    return urlparse(url).netloc.lower()


def is_same_or_subdomain(candidate: str, base: str) -> bool:
    if candidate == base:
        return True
    return candidate.endswith("." + base)


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def ethiopic_ratio(text: str) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    total_letters = sum(1 for ch in cleaned if ch.isalpha())
    if total_letters == 0:
        return 0.0
    eth_count = len(ETHIOPIC_RE.findall(cleaned))
    return eth_count / total_letters


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = normalize_text(text)
    parts = SENTENCE_SPLIT_RE.split(text)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def is_amharic_sentence(text: str, min_chars: int, min_eth_ratio: float) -> bool:
    if len(text) < min_chars:
        return False
    if ethiopic_ratio(text) < min_eth_ratio:
        return False

    language: str | None = None
    try:
        language = detect(text)
    except LangDetectException:
        language = None

    # langdetect frequently fails with "No features in text" on Ethiopic script.
    # Keep langdetect as primary detector, but fall back to langid when available.
    if language is None and langid is not None:
        try:
            language, _ = langid.classify(text)
        except Exception:
            language = None

    return language == "am"


def sentence_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StateDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visited_urls (
                url TEXT PRIMARY KEY,
                last_visited INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentence_hashes (
                hash TEXT PRIMARY KEY
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def should_visit(self, url: str, now_ts: int, revisit_after_seconds: int) -> bool:
        row = self.conn.execute("SELECT last_visited FROM visited_urls WHERE url = ?", (url,)).fetchone()
        if row is None:
            return True
        last_visited = int(row[0])
        return (now_ts - last_visited) >= revisit_after_seconds

    def mark_visited(self, url: str, now_ts: int) -> None:
        self.conn.execute(
            """
            INSERT INTO visited_urls (url, last_visited)
            VALUES (?, ?)
            ON CONFLICT(url) DO UPDATE SET last_visited=excluded.last_visited
            """,
            (url, now_ts),
        )

    def has_sentence_hash(self, digest: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM sentence_hashes WHERE hash = ? LIMIT 1", (digest,)).fetchone()
        return row is not None

    def add_sentence_hash(self, digest: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO sentence_hashes (hash) VALUES (?)", (digest,))

    def commit(self) -> None:
        self.conn.commit()


def build_session(cfg: ScraperConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent, "Accept-Language": "am,en;q=0.6"})

    retry = requests.adapters.Retry(
        total=cfg.request_retries,
        connect=cfg.request_retries,
        read=cfg.request_retries,
        backoff_factor=cfg.request_backoff_seconds,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def can_fetch_url(
    url: str,
    user_agent: str,
    parser_cache: dict[str, robotparser.RobotFileParser | None],
) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    parser = parser_cache.get(base)
    if parser is None and base not in parser_cache:
        parser = robotparser.RobotFileParser()
        parser.set_url(base + "/robots.txt")
        try:
            parser.read()
        except Exception:
            parser_cache[base] = None
            return True
        parser_cache[base] = parser

    parser = parser_cache.get(base)
    if parser is None:
        return True
    return parser.can_fetch(user_agent, url)


def extract_page_text_and_links(html: str, page_url: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form", "button", "input"]):
        tag.decompose()

    body = soup.body or soup
    text = body.get_text(separator=" ", strip=True)
    text = normalize_text(text)

    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href", "")).strip()
        if not href:
            continue
        joined = urljoin(page_url, href)
        canonical = canonicalize_url(joined)
        if canonical:
            links.append(canonical)

    return text, links


def allowed_link(
    url: str,
    *,
    cfg: ScraperConfig,
    seed_domains: set[str],
) -> bool:
    domain = domain_from_url(url)
    if not domain:
        return False

    allowlist = set(cfg.allowed_domains) if cfg.allowed_domains else set(seed_domains)
    if not cfg.same_domain_only:
        return True if not cfg.allowed_domains else domain in allowlist

    if cfg.allow_subdomains:
        return any(is_same_or_subdomain(domain, base) for base in allowlist)
    return domain in allowlist


def write_sentences(
    output_path: Path,
    state_db: StateDB,
    url: str,
    sentences: list[str],
) -> int:
    if not sentences:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(UTC).isoformat()
    written = 0

    with output_path.open("a", encoding="utf-8") as f:
        for sentence in sentences:
            digest = sentence_hash(sentence)
            if state_db.has_sentence_hash(digest):
                continue
            row = {"text": sentence, "url": url, "timestamp": now_iso}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            state_db.add_sentence_hash(digest)
            written += 1

    return written


def crawl(cfg: ScraperConfig, once: bool, max_pages_override: int | None) -> None:
    seeds = [canonicalize_url(s) for s in read_seeds(cfg.seeds_file)]
    seeds = [s for s in seeds if s]
    seed_domains = {domain_from_url(url) for url in seeds}

    queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
    queued_urls: set[str] = set(seeds)

    session = build_session(cfg)
    robots_cache: dict[str, robotparser.RobotFileParser | None] = {}
    state = StateDB(cfg.state_db)

    max_pages = max_pages_override if max_pages_override is not None else cfg.max_pages_per_cycle
    logging.info("Loaded %d seeds, max_pages_per_cycle=%d", len(seeds), max_pages)

    try:
        while True:
            pages_processed = 0
            sentences_written = 0
            cycle_start = time.time()

            while queue and pages_processed < max_pages:
                url, depth = queue.popleft()
                queued_urls.discard(url)

                if depth > cfg.max_depth:
                    continue

                if not allowed_link(url, cfg=cfg, seed_domains=seed_domains):
                    continue

                now_ts = int(time.time())
                if not state.should_visit(url, now_ts, cfg.revisit_after_seconds):
                    continue

                if cfg.respect_robots_txt and not can_fetch_url(url, cfg.user_agent, robots_cache):
                    continue

                try:
                    response = session.get(
                        url,
                        timeout=cfg.request_timeout_seconds,
                        verify=cfg.verify_ssl,
                    )
                    content_type = str(response.headers.get("Content-Type", "")).lower()
                    if response.status_code >= 400 or "text/html" not in content_type:
                        state.mark_visited(url, now_ts)
                        pages_processed += 1
                        continue
                    html = response.text
                except Exception:
                    state.mark_visited(url, now_ts)
                    pages_processed += 1
                    continue

                state.mark_visited(url, now_ts)
                pages_processed += 1

                text, links = extract_page_text_and_links(html, url)
                candidates = split_sentences(text)
                am_sentences = [
                    sentence
                    for sentence in candidates
                    if is_amharic_sentence(
                        sentence,
                        min_chars=cfg.min_sentence_chars,
                        min_eth_ratio=cfg.min_ethiopic_ratio,
                    )
                ]
                sentences_written += write_sentences(cfg.output_jsonl, state, url, am_sentences)

                if depth < cfg.max_depth:
                    for link in links:
                        if len(queue) >= cfg.max_queue_size:
                            break
                        if link in queued_urls:
                            continue
                        if not allowed_link(link, cfg=cfg, seed_domains=seed_domains):
                            continue
                        queue.append((link, depth + 1))
                        queued_urls.add(link)

                time.sleep(cfg.crawl_delay_seconds)

            state.commit()
            elapsed = time.time() - cycle_start
            logging.info(
                "Cycle done: pages=%d sentences_written=%d queue=%d elapsed=%.1fs",
                pages_processed,
                sentences_written,
                len(queue),
                elapsed,
            )

            if once:
                break

            if not queue:
                for seed in seeds:
                    if seed not in queued_urls:
                        queue.append((seed, 0))
                        queued_urls.add(seed)

            time.sleep(cfg.sleep_between_cycles_seconds)
    finally:
        state.commit()
        state.close()
        session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously scrape Amharic sentences into am.jsonl.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data_gathering/config.yaml"),
        help="Path to scraper YAML config.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single crawl cycle and exit.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Override max_pages_per_cycle for this run.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    logging.info("Writing output to %s", cfg.output_jsonl)
    crawl(cfg, once=args.once, max_pages_override=args.max_pages)


if __name__ == "__main__":
    main()
