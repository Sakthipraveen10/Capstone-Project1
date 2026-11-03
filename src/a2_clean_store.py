"""Part A tasks 3-5 (cleaning part): clean customers and orders, write data/clean/*.csv and data/instamart.db."""
import sqlite3
import numpy as np
import pandas as pd

import os
R, C = "data/raw/", "data/clean/"
os.makedirs(C, exist_ok=True)

# ------------------------------------------------------------------ customers
cust = pd.read_csv(R + "customers.csv")
products = pd.read_csv(R + "products.csv")

cust["city"] = cust["city"].str.strip().str.title()                              # (a)
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}          # (b)
g = cust["gender"].fillna("").str.strip().str.lower()
cust["gender"] = g.map(gmap).fillna("unknown")
bad_age = cust["age"].notna() & ((cust["age"] < 10) | (cust["age"] > 100))        # (c)
n_bad_age, n_blank_age = int(bad_age.sum()), int(cust["age"].isna().sum())
cust.loc[bad_age, "age"] = np.nan
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["age"] = cust["age"].round().astype(int)       # medians of integer ages may end in .5; round for a whole-year age
cust["signup_date"] = pd.to_datetime(cust["signup_date"])                         # (d)
print("customers: ages set to missing (out of range):", n_bad_age, "| blank ages:", n_blank_age,
      "| ages filled:", n_bad_age + n_blank_age)
print("cities:", cust["city"].value_counts().to_dict())
print("genders:", cust["gender"].value_counts().to_dict())
print("missing ages:", int(cust["age"].isna().sum()), "| age range:", cust["age"].min(), "-", cust["age"].max())

# ------------------------------------------------------------------ orders
o = pd.read_csv(R + "orders.csv", dtype={"order_amount": str}, keep_default_na=False,
                na_values={"delivery_minutes": [""]})
n0 = len(o)
o["order_ts"] = pd.to_datetime(o["order_ts"])                                    # rule 1
o = o.drop_duplicates()                                                          # rule 2
n_exact = n0 - len(o)
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")   # rule 3
n_latest = n1 - len(o)
o = o.merge(products[["product_id", "price_inr", "category"]], on="product_id", how="left")   # rule 4
amt = pd.to_numeric(o["order_amount"], errors="coerce")
n_unparse = int(amt.isna().sum())
calc = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = amt.fillna(calc)
neg = o["delivery_minutes"] < 0                                                  # rule 5
n_neg = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
cat_med = o.loc[comp].groupby("category")["delivery_minutes"].transform("median")
need = comp & o["delivery_minutes"].isna()
n_deliv_fill = int(need.sum())
o.loc[need, "delivery_minutes"] = cat_med[need]
o = o.drop(columns=["price_inr", "category"]).sort_values("order_ts").reset_index(drop=True)
print("\norders: raw", n0, "| exact duplicates removed", n_exact, "| superseded (earlier) records removed", n_latest,
      "| unparseable amounts filled", n_unparse, "| negative delivery set missing", n_neg,
      "| completed delivery_minutes imputed (incl. negatives)", n_deliv_fill)
print("rows:", len(o), "| status:", o["status"].value_counts().to_dict())
print("missing amount:", int(o["order_amount"].isna().sum()),
      "| missing delivery completed:", int(o.loc[o.status == "completed", "delivery_minutes"].isna().sum()),
      "| missing delivery cancelled:", int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()))
print("sum completed order_amount: INR", f"{o.loc[o.status == 'completed', 'order_amount'].sum():,.2f}")

# ------------------------------------------------------------------ store
o_out = o.copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = cust.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv(C + "customers.csv", index=False)
products.to_csv(C + "products.csv", index=False)
o_out.to_csv(C + "orders.csv", index=False)
with sqlite3.connect("data/instamart.db") as con:
    for name, df in [("customers", c_out), ("products", products), ("orders", o_out)]:
        df.to_sql(name, con, if_exists="replace", index=False)
print("\nwritten: data/clean/*.csv and data/instamart.db")
