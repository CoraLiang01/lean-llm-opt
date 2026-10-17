**Mathematical Model**

Let  

* \(P=\{p\mid p\text{ is a widget in the product list}\}\)  
* \(x_p\) – production quantity of widget \(p\) (continuous, \(x_p\ge 0\))  
* \(y\) – kilograms of CatalystX sold (continuous, \(y\ge 0\))

Parameters (to be read from the CSV files)

| Symbol | Definition | Source |
|--------|------------|--------|
| \(\text{LaborHours}_p\) | labor hours per unit of widget \(p\) | `product_resources.csv` column **LaborHours** |
| \(\text{MaterialA}_p\) | kg of Material A per unit of widget \(p\) | `product_resources.csv` column **MaterialA** |
| \(\text{MaterialB}_p\) | kg of Material B per unit of widget \(p\) | `product_resources.csv` column **MaterialB** |
| \(\text{Profit}_p\) | base profit per unit of widget \(p\) | `product_resources.csv` column **Profit** |
| \(L_{\max}\) | monthly labor‑hour limit | `resource_limits.csv` row *LaborHours* |
| \(A_{\max}\) | monthly Material A limit | `resource_limits.csv` row *MaterialA* |
| \(B_{\max}\) | monthly Material B limit | `resource_limits.csv` row *MaterialB* |

Constants  

* \(c_{\text{gen}} = 5\) kg CatalystX generated per unit of **Widget3**  
* \(p_{\text{sell}} = 300\) $ per kg CatalystX sold  
* \(p_{\text{dispose}} = 200\) $ per kg CatalystX disposed  

---

### Objective

\[
\max \;\; \sum_{p\in P}\text{Profit}_p\,x_p \;+\; p_{\text{sell}}\,y \;-\; p_{\text{dispose}}\bigl(c_{\text{gen}}\,x_{\text{Widget3}}-y\bigr)
\]

(Equivalently: \(\displaystyle \max \sum_{p}\text{Profit}_p\,x_p \;-\;1000\,x_{\text{Widget3}} \;+\;500\,y\))

---

### Constraints

1. **Resource limits**

\[
\begin{aligned}
\sum_{p\in P}\text{LaborHours}_p\,x_p &\le L_{\max} \\
\sum_{p\in P}\text{MaterialA}_p\,x_p &\le A_{\max} \\
\sum_{p\in P}\text{MaterialB}_p\,x_p &\le B_{\max}
\end{aligned}
\]

2. **CatalystX production / sale**

\[
\begin{aligned}
y &\le c_{\text{gen}}\,x_{\text{Widget3}} \\
y &\le 1500 \\
y &\ge 0
\end{aligned}
\]

3. **Production non‑negativity**

\[
x_p \ge 0 \quad \forall p\in P
\]

---

### Data Mapping Rules

```json
{
  "parameters": {
    "LaborHours": {
      "source_table": "file_1_view_0",
      "index_column": "Product",
      "value_column": "LaborHours"
    },
    "MaterialA": {
      "source_table": "file_1_view_0",
      "index_column": "Product",
      "value_column": "MaterialA"
    },
    "MaterialB": {
      "source_table": "file_1_view_0",
      "index_column": "Product",
      "value_column": "MaterialB"
    },
    "Profit": {
      "source_table": "file_1_view_0",
      "index_column": "Product",
      "value_column": "Profit"
    },
    "L_max": {
      "source_table": "file_0_view_0",
      "index_column": "Resource",
      "value_column": "MonthlyLimit",
      "filter": {"Resource":"LaborHours"}
    },
    "A_max": {
      "source_table": "file_0_view_0",
      "index_column": "Resource",
      "value_column": "MonthlyLimit",
      "filter": {"Resource":"MaterialA"}
    },
    "B_max": {
      "source_table": "file_0_view_0",
      "index_column": "Resource",
      "value_column": "MonthlyLimit",
      "filter": {"Resource":"MaterialB"}
    }
  },
  "constants": {
    "c_gen": 5,
    "p_sell": 300,
    "p_dispose": 200
  }
}
```

*The index set \(P\) is the set of all `Product` values returned from `file_1_view_0`.  
The special widget **Widget3** is identified by the string `"Widget3"` in that set.*