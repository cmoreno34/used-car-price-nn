"""A synthetic Spanish used-car market with a known pricing rule.

Why synthetic: there is no openly licensed dataset of Spanish second-hand
listings. The public ones are scraped from portals whose terms forbid it. This
generator reproduces the structure of a Spanish listing (brand, model segment,
registration year, km, power in CV, fuel, gearbox, DGT environmental label,
province, seller) with prices from an explicit rule, so a class can check what a
model learns against the truth.

The rule is deliberately NOT linear, which is the point of comparing a neural
network with a linear regression:
  - depreciation is exponential in age and faster for premium brands and EVs;
  - the km penalty saturates (the first 100,000 km cost more than the next);
  - diesel with a B label (or no label) loses value in the big cities
    (low-emission zones in Madrid and Barcelona) — an interaction;
  - automatic gearboxes are worth more in larger cars — another interaction.

    python scraper/synthetic_spain.py --n 6000 --seed 7 --out public/data/spain_synthetic.csv
"""
import argparse, csv, math, random

# Brand: (list price of a typical new compact, €; premium?)
BRANDS = {
    "Dacia": (15500, False), "Fiat": (18000, False), "Opel": (21500, False), "Citroën": (21500, False),
    "Renault": (22000, False), "Peugeot": (23500, False), "Seat": (23000, False), "Hyundai": (24000, False),
    "Kia": (24500, False), "Ford": (24000, False), "Nissan": (24500, False), "Skoda": (25000, False),
    "Toyota": (27000, False), "Mazda": (27500, False), "Volkswagen": (28500, False),
    "Volvo": (41000, True), "Audi": (43000, True), "BMW": (45000, True), "Mercedes-Benz": (47000, True),
    "Tesla": (48000, True),
}
BRAND_SHARE = {"Seat": 9, "Volkswagen": 9, "Renault": 8, "Peugeot": 8, "Toyota": 8, "Ford": 6, "Opel": 6,
               "Citroën": 6, "Kia": 6, "Hyundai": 6, "Dacia": 5, "Fiat": 4, "Nissan": 4, "Skoda": 4, "Mazda": 2,
               "BMW": 5, "Mercedes-Benz": 5, "Audi": 5, "Volvo": 2, "Tesla": 1}
# Segment: (price multiplier over a compact, typical CV)
SEGMENTS = {"city car": (0.72, 80), "compact": (1.0, 115), "sedan": (1.22, 150), "SUV": (1.28, 140), "MPV": (1.12, 130)}
SEG_SHARE = {"city car": 22, "compact": 30, "sedan": 12, "SUV": 30, "MPV": 6}
PROVINCES = {"Madrid": 16, "Barcelona": 14, "Valencia": 6, "Sevilla": 5, "Málaga": 5, "Alicante": 4, "Murcia": 3,
             "Vizcaya": 3, "Zaragoza": 3, "A Coruña": 3, "Asturias": 2, "Baleares": 3, "Las Palmas": 3,
             "Granada": 2, "Valladolid": 2, "Navarra": 2, "Other": 24}
BIG_CITIES = {"Madrid", "Barcelona"}


def pick(rng, weights):
    tot = sum(weights.values())
    u = rng.random() * tot
    for k, w in weights.items():
        u -= w
        if u <= 0:
            return k
    return k


def fuel_for(rng, year, brand):
    if brand == "Tesla":
        return "electric"
    w = {"petrol": 40, "diesel": 45 if year < 2016 else 22, "hybrid": 3 if year < 2015 else 18,
         "plug-in hybrid": 0 if year < 2017 else 5, "electric": 0 if year < 2016 else 4}
    if brand == "Toyota":
        w["hybrid"] *= 3
    if brand == "Dacia":
        w["electric"] = 1 if year >= 2021 else 0
    return pick(rng, w)


def dgt_label(fuel, year):
    """The DGT environmental label, from fuel and registration year."""
    if fuel == "electric" or fuel == "plug-in hybrid":
        return "0"
    if fuel == "hybrid":
        return "ECO"
    if fuel == "petrol":
        return "C" if year >= 2006 else "B" if year >= 2001 else "none"
    return "C" if year >= 2015 else "B" if year >= 2006 else "none"


def price_rule(brand, seg, year, km, cv, fuel, gearbox, label, province, seller, ref_year=2026):
    """The TRUE price (before noise). Documented in the README."""
    base, premium = BRANDS[brand]
    mult, cv_typ = SEGMENTS[seg]
    new_price = base * mult * (cv / cv_typ) ** 0.55
    new_price *= {"petrol": 1.0, "diesel": 1.05, "hybrid": 1.10, "plug-in hybrid": 1.30, "electric": 1.35}[fuel]
    age = ref_year - year + 0.5
    rate = 0.11 + (0.035 if premium else 0.0) + (0.04 if fuel == "electric" else 0.0)
    value = new_price * math.exp(-rate * age)
    value *= 1.0 - 0.28 * (1 - math.exp(-km / 110000))              # saturating km penalty
    if fuel == "diesel" and label in ("B", "none") and province in BIG_CITIES:
        value *= 0.78                                               # low-emission zones
    if label == "none":
        value *= 0.85
    if gearbox == "automatic":
        value *= 1.0 + (0.10 if seg in ("sedan", "SUV") or premium else 0.04)
    if seller == "dealer":
        value *= 1.08
    return max(value, 900.0)


def generate(n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        brand = pick(rng, BRAND_SHARE)
        seg = pick(rng, SEG_SHARE)
        if brand == "Tesla":
            seg = rng.choice(["sedan", "SUV"])
        year = min(2025, max(2006, int(round(2026 - rng.gammavariate(2.2, 3.2)))))
        age = 2026 - year
        km = int(max(500, rng.lognormvariate(math.log(max(1, age) * 15000), 0.35)))
        km = min(km, 420000) // 100 * 100
        cv_typ = SEGMENTS[seg][1] * (1.35 if BRANDS[brand][0] > 40000 else 1.0)
        cv = int(round(cv_typ * rng.lognormvariate(0, 0.18) / 5) * 5)
        fuel = fuel_for(rng, year, brand)
        premium = BRANDS[brand][1]
        gearbox = "automatic" if rng.random() < (0.85 if premium or fuel in ("electric", "hybrid", "plug-in hybrid") else 0.25 + 0.02 * (year - 2006)) else "manual"
        if fuel == "electric":
            gearbox = "automatic"
        label = dgt_label(fuel, year)
        province = pick(rng, PROVINCES)
        seller = "dealer" if rng.random() < 0.62 else "private"
        true = price_rule(brand, seg, year, km, cv, fuel, gearbox, label, province, seller)
        price = int(round(true * rng.lognormvariate(0, 0.10) / 50) * 50)
        rows.append({"brand": brand, "segment": seg, "year": year, "km": km, "power_cv": cv, "fuel": fuel,
                     "gearbox": gearbox, "dgt_label": label, "province": province, "seller": seller, "price_eur": price})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="public/data/spain_synthetic.csv")
    a = ap.parse_args()
    rows = generate(a.n, a.seed)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cars -> {a.out}")
