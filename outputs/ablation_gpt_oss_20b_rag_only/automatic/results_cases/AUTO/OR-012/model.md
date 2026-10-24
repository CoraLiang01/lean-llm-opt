**Mathematical Model**

```
SETS
    I   – set of products (identified by “Product Name”)

PARAMETERS
    Revenue[i]      – revenue per unit of product i
    Demand[i]       – deterministic demand for product i
    InitInv[i]      – initial inventory of product i

VARIABLES
    x[i]  ≥ 0        – units of product i to sell  (integer)

OBJECTIVE
    maximize  Z = Σ_{i∈I} Revenue[i] * x[i]

CONSTRAINTS
    1.  x[i] ≤ Demand[i]          ∀ i∈I
    2.  x[i] ≤ InitInv[i]         ∀ i∈I
```

*All variables `x[i]` are non‑negative integers.*

---

**Data Mapping (CSVQA_DATA)**  

| Symbol | Table ID          | Column Name          |
|--------|-------------------|----------------------|
| I      | file_0_view_0     | Product Name         |
| Revenue[i] | file_0_view_0 | Revenue              |
| Demand[i]   | file_0_view_0 | Demand               |
| InitInv[i]  | file_0_view_0 | Initial Inventory    |

*The mapping uses the exact table identifier and column names returned by CSVQA. No sorting or re‑indexing is performed; the original row order is preserved.*