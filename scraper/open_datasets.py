"""Downloads an openly licensed used-car dataset and writes the sample the app ships.

Source: "Craigslist Used Cars and Trucks: EDA" (Hugging Face, Yoad22, CC-BY-4.0),
a cleaned version of Austin Reese's Craigslist scrape (Kaggle). US listings,
prices in US dollars, odometer in miles. The app ships a reproducible random
sample so the page stays light; run this with a larger --n for more.

    python scraper/open_datasets.py --n 8000 --seed 11 --out public/data/us_craigslist_sample.csv
"""
import argparse, io
import pandas as pd
import requests

URL = "https://huggingface.co/datasets/Yoad22/craigslist-used-cars-eda/resolve/main/vehicles_clean.csv"
KEEP = ["price", "year", "manufacturer", "condition", "cylinders", "fuel", "odometer", "transmission",
        "drive", "type", "paint_color", "state"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="public/data/us_craigslist_sample.csv")
    ap.add_argument("--local", help="use an already-downloaded vehicles_clean.csv")
    a = ap.parse_args()
    if a.local:
        d = pd.read_csv(a.local)
    else:
        r = requests.get(URL, timeout=120, headers={"User-Agent": "used-car-price-nn (course material)"})
        r.raise_for_status()
        d = pd.read_csv(io.BytesIO(r.content))
    d = d[KEEP]
    # Listings that are almost certainly placeholders or typos.
    d = d[(d.odometer > 100) & (d.odometer < 400000) & (d.year >= 1995) & (d.price >= 1000)]
    d = d.sample(n=min(a.n, len(d)), random_state=a.seed).sort_values("price", ignore_index=True)
    d["year"] = d.year.astype(int)
    d["odometer"] = d.odometer.astype(int)
    d["cylinders"] = d.cylinders.astype(int)
    d = d.rename(columns={"price": "price_usd", "odometer": "odometer_miles"})
    d.to_csv(a.out, index=False)
    print(f"{len(d)} listings -> {a.out}")


if __name__ == "__main__":
    main()
