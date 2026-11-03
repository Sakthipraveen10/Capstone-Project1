"""Part A task 7: pandas cross-check of Q2 and exploratory statistics of completed order_amount."""
import pandas as pd

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
d = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")

pt = pd.pivot_table(d, index="city", columns="category", values="order_amount", aggfunc="sum", margins=True)
print("pivot grand total:", f"{pt.loc['All', 'All']:,.2f}", "(SQL Q2 total 2,998,668.80)")
print(pt.round(2).to_string())

agg = d.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), n=("order_amount", "count")).round(2)
print("\nper city sum / mean / count:\n", agg.to_string())

x = d["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {x.mean():.2f} median {x.median():.2f} skew {x.skew():.2f} Q1 {q1:.2f} Q3 {q3:.2f}")
print(f"fences low {lo:.2f} high {hi:.2f} | outliers below {int((x < lo).sum())} above {int((x > hi).sum())}")
corr = d[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\ncorrelation matrix:\n", corr.round(3).to_string())
import itertools
pairs = {(a, b): corr.loc[a, b] for a, b in itertools.combinations(corr.columns, 2)}
best = max(pairs, key=lambda k: abs(pairs[k]))
print("largest |corr| pair:", best, round(pairs[best], 3))
