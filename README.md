# What is my used car worth? — a configurable neural network

**Live:** https://cmoreno34.github.io/used-car-price-nn/

A neural network learns second-hand car prices from listings and then prices a car you describe. Everything is
configurable in the browser — the data, the variables, the layers, the activations, dropout, L2, the optimiser — and
it always reports against a **linear regression on the same variables** and a **mean baseline**, on cars it never saw.
Built for the Marketing Analytics course at Universidad Francisco de Vitoria.

## The page

1. **Data** — a synthetic Spanish market, real US listings, or your own CSV (for example, from the scraper below).
2. **Variables** — the price column, the price or its logarithm as target, the features. Numeric features are
   standardised and categorical ones one-hot encoded, fitted on the training cars only; levels with fewer than 10
   training cars are pooled as “other”.
3. **Network** — presets (none / 16 / 64→32 / 128→64→32) or any stack of layers; the page counts the weights and
   warns when there are more weights than cars.
4. **Train** — TensorFlow.js on the CPU backend (a few seconds; identical numbers on every laptop). Live training and
   validation curves; early stopping restores the weights of the best validation epoch.
5. **Results** — MAE, RMSE, MAPE and R² on the held-out cars; predicted-vs-actual for both models; permutation
   importance of each original variable.
6. **Price my car** — the network's and the regression's price, a range taken from the test errors (the band that
   held 80 % of test cars), and the six most similar cars in the data.

The configuration travels in the URL (*Copy a link to this configuration*), so a class can open the same network.

## Data and licences

| File | What it is | Licence |
|---|---|---|
| `public/data/spain_synthetic.csv` | 6,000 synthetic Spanish listings from `scraper/synthetic_spain.py` (documented rule below) | this repository |
| `public/data/us_craigslist_sample.csv` | 8,000 real US listings sampled by `scraper/open_datasets.py` from *Craigslist Used Cars and Trucks: EDA* (Hugging Face, Yoad22), itself from Austin Reese's Kaggle scrape of Craigslist | CC-BY-4.0 — attribute the source |

**Why a synthetic Spanish market.** There is no openly licensed dataset of Spanish second-hand listings: the public ones
are scraped from portals whose terms forbid it. The generator reproduces a Spanish listing (brand, segment, year, km,
CV, fuel, gearbox, DGT label, province, seller) with prices from an explicit, deliberately non-linear rule:
exponential depreciation, faster for premium brands and electric cars; a km penalty that saturates; diesel cars with a
B label (or none) losing 22 % in Madrid and Barcelona (low-emission zones); automatic gearboxes worth more in larger
cars; 10 % noise. Because the rule is known, a class can check what the network learned. A linear regression on the
raw price reaches ≈ 45 % MAPE on it; on the log price ≈ 10 %; a 64→32 network ≈ 9–11 % on the raw price — it finds by
itself what the linear model needs the logarithm for.

## The scraper (`scraper/`)

```bash
pip install -r scraper/requirements.txt
python scraper/scrape.py scraper/sources/example_local.yaml --out my_cars.csv   # runs on two local sample pages
```

A source is a YAML file with CSS selectors for a listing card and its fields (see `sources/example_local.yaml`).
The scraper is deliberately polite and will refuse to:

- fetch anything `robots.txt` disallows, or run faster than one request every 3 s (or the site's crawl-delay);
- run against a site unless the config says `terms_checked: true` with the `terms_url` you read;
- hide what it is (it sends a user agent naming this project);
- retry around a block — a 403, 429, 503 or CAPTCHA page ends the run.

**Do not configure it for coches.net, milanuncios, Wallapop or similar portals**: their terms of use forbid automated
collection, whatever their robots.txt says. Use it on sites that allow it, on your own organisation's pages, or on
data you are given.

`scraper/open_datasets.py` rebuilds the US sample; `scraper/synthetic_spain.py --n --seed` regenerates the Spanish one.

## Development

```bash
npm install
npm run dev        # http://localhost:5173/used-car-price-nn/
npm test           # linear baseline + metrics vs NumPy/scikit-learn; a network must beat linear regression
python test/ref_models.py   # regenerates the reference values
```

César Moreno Pascual, PhD — Marketing Analytics, Universidad Francisco de Vitoria.
