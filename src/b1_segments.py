"""Part B tasks 1-3: RFM table, K-means selection, segment profile/naming, PCA, outputs/customer_segments.csv."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import os
os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
d = o[o.status == "completed"]

rfm = d.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                   monetary=("order_amount", "sum"))
rfm["recency"] = (SNAPSHOT - rfm["last"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
print("customers:", len(rfm), "(ref 1175)")
print("recency median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max(), "(ref 19/0/180)")
print("frequency mean/max:", round(rfm.frequency.mean(), 2), rfm.frequency.max(), "(ref 9.46/33)")
print("monetary mean/max:", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2), "(ref 2552.06/14766.55)")

X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\nK  inertia   silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"{k}  {km.inertia_:8.1f}  {silhouette_score(X, km.labels_):.3f}")

km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2)

# Names are assigned by rule from the profile (cluster numbers themselves carry no meaning).
names, left = {}, set(prof.index)
premium = prof["mean_monetary"].idxmax(); names[premium] = "Premium Big Spenders"; left.discard(premium)
loyal = prof.loc[list(left), "mean_frequency"].idxmax(); names[loyal] = "Loyal Frequent Buyers"; left.discard(loyal)
lapsing = prof.loc[list(left), "mean_recency"].idxmax(); names[lapsing] = "Fading Occasional Buyers"; left.discard(lapsing)
for s in left:
    names[s] = "Light Regulars"
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium
prof["segment_name"] = prof.index.map(names)
print("\nsegment profile (K=%d):\n" % FINAL_K, prof.to_string())

pca = PCA(n_components=2).fit(X)
print("\nPCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)
print("premium segment id:", premium, "| premium customers:", int(rfm.is_premium.sum()))
