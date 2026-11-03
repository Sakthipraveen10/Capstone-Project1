"""Part A task 5: assert reference checkpoints from the audit and the cleaning; print ALL CHECKPOINTS PASSED."""
import sqlite3
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
done = o[o.status == "completed"]
checks = {
    "orders rows == 12309": len(o) == 12309,
    "order_id unique": o.order_id.is_unique,
    "completed == 11110": len(done) == 11110,
    "cancelled == 1199": int((o.status == "cancelled").sum()) == 1199,
    "no pending rows": int((o.status == "pending").sum()) == 0,
    "0 missing order_amount": int(o.order_amount.isna().sum()) == 0,
    "0 missing delivery (completed)": int(done.delivery_minutes.isna().sum()) == 0,
    "1199 missing delivery (cancelled)": int(o[o.status == "cancelled"].delivery_minutes.isna().sum()) == 1199,
    "completed revenue == 2998668.80": round(done.order_amount.sum(), 2) == 2998668.80,
    "exactly 6 cities": c.city.nunique() == 6,
    "city counts": c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "genders 610/560/30": c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages, all in 18-60": int(c.age.isna().sum()) == 0 and c.age.between(18, 60).all(),
}
with sqlite3.connect("data/instamart.db") as con:
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    checks["db has 3 tables"] = {"customers", "products", "orders"} <= tables
    checks["db orders == 12309"] = con.execute("select count(*) from orders").fetchone()[0] == 12309
for name, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + name)
assert all(checks.values()), "a checkpoint failed"
print("ALL CHECKPOINTS PASSED")
