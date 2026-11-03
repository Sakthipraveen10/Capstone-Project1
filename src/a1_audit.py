"""Part A task 2: audit the raw tables."""
import pandas as pd

R = "data/raw/"
cust = pd.read_csv(R + "customers.csv")
prod = pd.read_csv(R + "products.csv")
# order_amount is read as text: pandas would silently turn the planted "N/A" into NaN
orders = pd.read_csv(R + "orders.csv", dtype={"order_amount": str}, keep_default_na=False, na_values={"order_amount": [], "delivery_minutes": [""]})

for name, df in [("customers", cust), ("products", prod), ("orders", orders)]:
    print(f"\n=== {name}: shape {df.shape}")
    summary = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                            "missing_pct": (df.isna().mean() * 100).round(2)})
    print(summary.to_string())

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
nonnumeric = int(amt_num.isna().sum())
deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(cust["age"], errors="coerce")
gender_raw = cust["gender"].dropna()
rows = [
    ("raw orders.csv rows", len(orders), 13084),
    ("exact duplicate rows", int(orders.duplicated().sum()), 119),
    ("rows with status pending", int((orders["status"] == "pending").sum()), 656),
    ("distinct order_id", int(orders["order_id"].nunique()), 12309),
    ("non-numeric order_amount", nonnumeric, 120),
    ("negative delivery_minutes", int((deliv < 0).sum()), 106),
    ("blank delivery_minutes", int(deliv.isna().sum()), 2541),
    ("customers: blank age", int(age.isna().sum()), 61),
    ("customers: age outside 10-100", int(((age < 10) | (age > 100)).sum()), 6),
    ("customers: distinct raw city spellings", int(cust["city"].nunique()), 24),
    ("customers: distinct non-blank gender spellings", int(gender_raw.nunique()), 10),
    ("customers: blank gender", int(cust["gender"].isna().sum()), 30),
]
print("\n=== Audit figures")
print(f"{'figure':50s}{'observed':>10s}{'reference':>10s}")
for n, v, ref in rows:
    print(f"{n:50s}{v:>10,}{ref:>10,}  {'OK' if v == ref else 'MISMATCH'}")
