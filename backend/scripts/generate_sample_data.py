"""Generate realistic sample data for the commodity forecaster demo.

Creates three CSV files in data/raw/:
  - onion_prices.csv: daily market prices across 5 Indian markets
  - onion_production.csv: monthly Maharashtra production estimates
  - fuel_prices.csv: daily diesel prices in Mumbai and Delhi
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _seasonal(day_of_year: np.ndarray) -> np.ndarray:
    """Onion seasonal price pattern: peaks Oct-Dec, dips Mar-May."""
    # Primary peak around day 300 (late Oct)
    peak1 = np.exp(-0.5 * ((day_of_year - 300) / 40) ** 2)
    # Secondary smaller peak around day 180 (late Jun, monsoon disruption)
    peak2 = 0.4 * np.exp(-0.5 * ((day_of_year - 180) / 30) ** 2)
    # Dip around day 110 (mid Apr, rabi harvest)
    dip = -0.5 * np.exp(-0.5 * ((day_of_year - 110) / 35) ** 2)
    return peak1 + peak2 + dip


def generate_onion_prices(rng: np.random.Generator) -> pd.DataFrame:
    """Generate daily onion prices for 5 markets from 2020-01-01 to 2025-09-30."""
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2025-09-30")
    all_dates = pd.date_range(start, end, freq="D")

    markets = {
        "Lasalgaon": ("Maharashtra", 0.0),    # base market
        "Azadpur": ("Delhi", 250.0),           # higher due to transport
        "Pimpalgaon": ("Maharashtra", -50.0),  # slightly lower
        "Bangalore": ("Karnataka", 200.0),     # moderate premium
        "Kolkata": ("West Bengal", 300.0),      # highest transport cost
    }

    rows = []
    # Generate a shared random-walk component for correlated markets
    n_days = len(all_dates)
    random_walk = np.cumsum(rng.normal(0, 3, n_days))
    # Clamp the walk so it doesn't drift too far
    random_walk = np.clip(random_walk, -600, 600)

    day_of_year = all_dates.dayofyear.values.astype(float)
    year_frac = (all_dates - start).days / 365.25

    # Base price trend: ~1500 + slight upward trend + seasonal + walk
    base_price = (
        1500
        + 40 * year_frac                     # slight upward trend
        + 800 * _seasonal(day_of_year)        # seasonal pattern
        + random_walk                         # regime shifts
    )

    for market, (state, premium) in markets.items():
        # Market-specific noise
        market_noise = rng.normal(0, 50, n_days)
        modal = base_price + premium + market_noise
        modal = np.clip(modal, 300, 8000).round(2)

        min_price = (modal * rng.uniform(0.85, 0.95, n_days)).round(2)
        max_price = (modal * rng.uniform(1.05, 1.15, n_days)).round(2)

        for i, date in enumerate(all_dates):
            # Skip ~30% of days (weekends + holidays + random gaps)
            dow = date.dayofweek
            if dow == 6:  # Always skip Sunday
                continue
            if dow == 5 and rng.random() < 0.7:  # Skip most Saturdays
                continue
            if rng.random() < 0.03:  # Random holiday/closure
                continue

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "market": market,
                "state": state,
                "variety": "Onion",
                "min_price": min_price[i],
                "max_price": max_price[i],
                "modal_price": modal[i],
                "unit": "INR/Quintal",
            })

    df = pd.DataFrame(rows)

    # Introduce a few nulls (about 0.2% of rows)
    null_indices = rng.choice(len(df), size=int(len(df) * 0.002), replace=False)
    for idx in null_indices:
        col = rng.choice(["min_price", "max_price", "modal_price"])
        df.loc[idx, col] = np.nan

    # Introduce a few negative prices to test validation (about 10 rows)
    neg_indices = rng.choice(len(df), size=10, replace=False)
    df.loc[neg_indices, "modal_price"] = -100.0

    df = df.sort_values(["date", "market"]).reset_index(drop=True)
    return df


def generate_onion_production(rng: np.random.Generator) -> pd.DataFrame:
    """Generate monthly Maharashtra onion production (2020-01 to 2025-09)."""
    dates = pd.date_range("2020-01-01", "2025-09-01", freq="MS")
    rows = []
    for date in dates:
        month = date.month
        # Rabi harvest peaks April-May, Kharif peaks Oct-Nov
        if month in (4, 5):
            base_prod = 1_200_000
        elif month in (10, 11):
            base_prod = 900_000
        elif month in (3, 6):
            base_prod = 800_000
        else:
            base_prod = 600_000

        # Year-over-year growth ~2%
        year_factor = 1 + 0.02 * (date.year - 2020)
        production = base_prod * year_factor * rng.uniform(0.85, 1.15)
        area = production / rng.uniform(12, 18)  # rough yield per hectare

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "state": "Maharashtra",
            "commodity": "Onion",
            "production_tonnes": round(production),
            "area_hectares": round(area),
        })

    return pd.DataFrame(rows)


def generate_fuel_prices(rng: np.random.Generator) -> pd.DataFrame:
    """Generate daily diesel prices for Mumbai and Delhi (2020-01-01 to 2025-09-30)."""
    dates = pd.date_range("2020-01-01", "2025-09-30", freq="D")
    n = len(dates)

    rows = []
    cities = {"Mumbai": 2.0, "Delhi": 0.0}

    # Diesel base: started ~70, rose to ~90 by mid-2022, then fluctuated
    day_index = np.arange(n, dtype=float)
    base = 70 + 15 * (1 / (1 + np.exp(-0.005 * (day_index - 600))))  # sigmoid rise
    walk = np.cumsum(rng.normal(0, 0.05, n))
    walk = np.clip(walk, -5, 5)
    base = base + walk

    for city, premium in cities.items():
        city_noise = rng.normal(0, 0.3, n)
        price = np.clip(base + premium + city_noise, 60, 100).round(2)

        for i, date in enumerate(dates):
            # Skip some days
            if date.dayofweek == 6:
                continue
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "fuel_type": "Diesel",
                "price_per_litre": price[i],
                "city": city,
            })

    return pd.DataFrame(rows).sort_values(["date", "city"]).reset_index(drop=True)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Generating onion_prices.csv ...")
    df_prices = generate_onion_prices(rng)
    df_prices.to_csv(RAW_DIR / "onion_prices.csv", index=False)
    print(f"  → {len(df_prices)} rows")

    print("Generating onion_production.csv ...")
    df_prod = generate_onion_production(rng)
    df_prod.to_csv(RAW_DIR / "onion_production.csv", index=False)
    print(f"  → {len(df_prod)} rows")

    print("Generating fuel_prices.csv ...")
    df_fuel = generate_fuel_prices(rng)
    df_fuel.to_csv(RAW_DIR / "fuel_prices.csv", index=False)
    print(f"  → {len(df_fuel)} rows")

    print("Done. Files written to", RAW_DIR)


if __name__ == "__main__":
    main()
