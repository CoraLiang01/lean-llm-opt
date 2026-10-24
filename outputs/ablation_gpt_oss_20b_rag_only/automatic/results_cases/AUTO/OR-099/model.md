**Mathematical model (facility‑location problem)**  

```
Sets
    I   – warehouses  (i = 1,…,11)
    J   – stores      (j = 1,…,11)

Parameters
    f[i]   – opening cost of warehouse i
    cap[i] – capacity of warehouse i
    d[j]   – demand of store j
    c[i,j] – transportation cost per unit from warehouse i to store j

Decision variables
    y[i]   ∈ {0,1}   – 1 if warehouse i is opened
    x[i,j] ≥ 0       – units shipped from i to j

Objective
    minimize   Σ_{i∈I} f[i] y[i]  +  Σ_{i∈I} Σ_{j∈J} c[i,j] x[i,j]

Constraints
    (1) Demand satisfaction
        Σ_{i∈I} x[i,j]  =  d[j]          ∀ j∈J

    (2) Capacity limits
        Σ_{j∈J} x[i,j]  ≤  cap[i] y[i]   ∀ i∈I

    (3) Non‑negativity
        x[i,j] ≥ 0                       ∀ i∈I, j∈J
```

---

**Data‑mapping rules**

| Parameter | Source table | Column name | Mapping description |
|-----------|--------------|-------------|---------------------|
| `f[i]` | `file_0_view_0` | `Opening Cost (fi)` | `i` is the value of `Warehouse (i)` |
| `cap[i]` | `file_0_view_0` | `Capacity (units)` | `i` is the value of `Warehouse (i)` |
| `d[j]` | `file_1_view_0` | `Demand (units, dj)` | `j` is the value of `Store (j)` |
| `c[i,j]` | `file_2_view_0` | matrix entry | row axis `Unnamed: 0` → store `j`; column axis `Warehouse (i)` → warehouse `i` |

All index sets `I` and `J` are the full sets of warehouse and store identifiers present in the respective tables (1–11). No additional enumeration of values is required; the model uses the symbolic parameters linked to the CSV data via the mapping table above.