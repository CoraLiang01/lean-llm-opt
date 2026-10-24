**Mathematical Model**

Let  

* \(P=\{ \text{Widget1},\dots ,\text{Widget141}\}\) – set of products.  
* \(x_p\) – production quantity of product \(p\in P\) (continuous, \(x_p\ge 0\)).  
* \(y\) – kilograms of CatalystX sold in the month (continuous, \(0\le y\le 1500\)).  

Parameters (all taken from the CSV files)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_p\) | Base profit per unit of product \(p\) | `product_resources.csv` – column **Profit** |
| \(l_p\) | Labor hours required per unit of product \(p\) | `product_resources.csv` – column **LaborHours** |
| \(a_p\) | Material A consumption per unit of product \(p\) | `product_resources.csv` – column **MaterialA** |
| \(b_p\) | Material B consumption per unit of product \(p\) | `product_resources.csv` – column **MaterialB** |
| \(L\) | Monthly labor‑hour limit | `resource_limits.csv` – row **LaborHours** |
| \(A\) | Monthly Material A limit | `resource_limits.csv` – row **MaterialA** |
| \(B\) | Monthly Material B limit | `resource_limits.csv` – row **MaterialB** |
| \(r\) | CatalystX production rate per unit of Widget3 | constant \(5\) kg/unit |
| \(s\) | Sale price of CatalystX | constant \(300\) $/kg |
| \(d\) | Disposal cost of unsold CatalystX | constant \(200\) $/kg |

---

### Objective

\[
\max \;\; \sum_{p\in P} c_p\,x_p \;+\; s\,y \;-\; d\,(r\,x_{\text{Widget3}}-y)
\]

(Equivalently \( \max \sum_{p} c_p x_p + 500\,y - 1000\,x_{\text{Widget3}}\).)

### Constraints

1. **Labor limit**

\[
\sum_{p\in P} l_p\,x_p \;\le\; L
\]

2. **Material A limit**

\[
\sum_{p\in P} a_p\,x_p \;\le\; A
\]

3. **Material B limit**

\[
\sum_{p\in P} b_p\,x_p \;\le\; B
\]

4. **CatalystX sales**

\[
0 \;\le\; y \;\le\; \min\{1500,\; r\,x_{\text{Widget3}}\}
\]

5. **Non‑negativity**

\[
x_p \;\ge\; 0 \quad \forall p\in P,\qquad y \;\ge\; 0
\]

---

### Data Mapping (exact source references)

```json
{
  "table_id": "file_1_view_0",
  "columns": {
    "Product": "Product",
    "LaborHours": "l_p",
    "MaterialA": "a_p",
    "MaterialB": "b_p",
    "Profit": "c_p"
  },
  "index_set": "P",
  "resource_limits": {
    "file_id": "file_0_view_0",
    "rows": {
      "LaborHours": {"L": "MonthlyLimit"},
      "MaterialA": {"A": "MonthlyLimit"},
      "MaterialB": {"B": "MonthlyLimit"}
    }
  },
  "catalyst": {
    "rate": 5,
    "sale_price": 300,
    "disposal_cost": 200,
    "max_sale": 1500
  }
}
```

All parameters are bound to the corresponding CSV columns; no numerical values are enumerated in the model. The model is ready for implementation in any optimization solver.