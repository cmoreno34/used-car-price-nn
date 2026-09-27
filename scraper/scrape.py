"""A polite, configurable listings scraper for building your own used-car dataset.

Each source is a YAML file (see sources/) that says where the listings are and
how to read one: CSS selectors for the card and for each field. The scraper
then walks the pages and writes one CSV row per car, ready for the app.

What it will not do, on purpose:
  - fetch anything robots.txt disallows for its user agent;
  - run against a site whose terms you have not confirmed allow it
    (`terms_checked: true` in the config, with the URL of the terms);
  - hide what it is: it sends a user agent that names the project;
  - retry around a block. A 403, a 429, a 503 or a CAPTCHA page ends the run.
Portals such as coches.net, milanuncios or Wallapop forbid automated
extraction in their terms, whatever robots.txt says; do not configure them.

    python scraper/scrape.py sources/example_local.yaml --out my_cars.csv
    python scraper/scrape.py sources/my_site.yaml --dry-run
"""
import argparse, csv, os, re, sys, time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

MIN_DELAY = 3.0
MAX_PAGES = 50
BLOCK_HINTS = re.compile(r"captcha|are you a robot|access denied|unusual traffic|verify you are human", re.I)


class Stop(Exception):
    pass


def load_config(path):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    for k in ("name", "start_url", "item", "fields"):
        if k not in cfg:
            raise SystemExit(f"{path}: missing '{k}'")
    local = cfg["start_url"].startswith("file:")
    if not local and not (cfg.get("terms_checked") is True and cfg.get("terms_url")):
        raise SystemExit(
            f"{path}: set terms_checked: true and terms_url: <the site's terms> once you have read them and "
            "they allow automated collection. Many listing portals forbid it even where robots.txt allows it.")
    cfg["delay_seconds"] = max(MIN_DELAY, float(cfg.get("delay_seconds", MIN_DELAY)))
    cfg["max_pages"] = min(MAX_PAGES, int(cfg.get("max_pages", 5)))
    cfg.setdefault("user_agent", "used-car-price-nn/1.0 (university course exercise; +https://github.com/cmoreno34/used-car-price-nn)")
    return cfg


class Fetcher:
    def __init__(self, cfg, base_dir):
        self.cfg, self.base_dir = cfg, base_dir
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg["user_agent"]
        self.robots, self.last = {}, 0.0

    def allowed(self, url):
        p = urlparse(url)
        if p.scheme == "file":
            return True
        root = f"{p.scheme}://{p.netloc}"
        if root not in self.robots:
            rp = urllib.robotparser.RobotFileParser(root + "/robots.txt")
            try:
                rp.read()
            except Exception:
                raise Stop(f"could not read {root}/robots.txt — not scraping a site whose rules are unknown")
            self.robots[root] = rp
            cd = rp.crawl_delay(self.cfg["user_agent"])
            if cd:
                self.cfg["delay_seconds"] = max(self.cfg["delay_seconds"], float(cd))
        return self.robots[root].can_fetch(self.cfg["user_agent"], url)

    def get(self, url):
        if not self.allowed(url):
            raise Stop(f"robots.txt disallows {url}")
        p = urlparse(url)
        if p.scheme == "file":
            path = os.path.join(self.base_dir, p.path.lstrip("/")) if not os.path.isabs(p.path.lstrip("/")) else p.path
            return open(path, encoding="utf-8").read()
        wait = self.last + self.cfg["delay_seconds"] - time.time()
        if wait > 0:
            time.sleep(wait)
        r = self.session.get(url, timeout=30)
        self.last = time.time()
        if r.status_code in (403, 429, 503):
            raise Stop(f"{r.status_code} from {url} — the site is refusing automated requests; stopping")
        r.raise_for_status()
        if BLOCK_HINTS.search(r.text[:20000]):
            raise Stop(f"{url} looks like a bot challenge; stopping rather than working around it")
        return r.text


def parse_number(s):
    """'12.990 €' -> 12990 ; '1,6' -> 1.6 ; '150.000 km' -> 150000 (Spanish thousands dots)."""
    s = (s or "").strip()
    m = re.search(r"-?\d[\d.,\s]*", s)
    if not m:
        return None
    t = m.group(0).replace(" ", "").replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", t):
        t = t.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
        t = t.replace(",", "")
    else:
        t = t.replace(",", ".")
    try:
        v = float(t)
        return int(v) if v.is_integer() else v
    except ValueError:
        return None


def read_field(card, spec):
    if isinstance(spec, str):
        spec = {"selector": spec}
    el = card.select_one(spec["selector"]) if spec.get("selector") else card
    if el is None:
        return None
    val = el.get(spec["attr"]) if spec.get("attr") else el.get_text(" ", strip=True)
    if spec.get("regex") and val is not None:
        m = re.search(spec["regex"], val)
        val = m.group(1) if m else None
    if spec.get("type") == "number":
        val = parse_number(val)
    return val


def scrape(cfg, base_dir, dry_run=False):
    f = Fetcher(cfg, base_dir)
    url, rows, seen = cfg["start_url"], [], set()
    for page in range(1, cfg["max_pages"] + 1):
        if not url or url in seen:
            break
        seen.add(url)
        if dry_run:
            print(f"would fetch page {page}: {url} (robots.txt allows: {f.allowed(url)}; delay {cfg['delay_seconds']}s)")
            break
        html = f.get(url)
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.select(cfg["item"])
        for c in cards:
            row = {k: read_field(c, spec) for k, spec in cfg["fields"].items()}
            if all(row.get(k) is not None for k in cfg.get("required", [])):
                rows.append(row)
        print(f"page {page}: {len(cards)} listings, {len(rows)} kept so far")
        nxt = soup.select_one(cfg["next_page"]) if cfg.get("next_page") else None
        url = urljoin(url, nxt["href"]) if nxt is not None and nxt.get("href") else None
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default="scraped_cars.csv")
    ap.add_argument("--dry-run", action="store_true", help="check robots.txt and show the first request only")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        rows = scrape(cfg, os.path.dirname(os.path.abspath(a.config)), a.dry_run)
    except Stop as e:
        print("STOPPED:", e)
        sys.exit(2)
    if a.dry_run:
        return
    if not rows:
        print("no listings matched the selectors — check 'item' and 'fields' in the config")
        sys.exit(1)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cfg["fields"]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cars -> {a.out}")


if __name__ == "__main__":
    main()
