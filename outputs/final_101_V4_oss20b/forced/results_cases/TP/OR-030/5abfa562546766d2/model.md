**Mathematical Model**

```
Sets
    I : index of FDK57 product records.

Parameters
    r_i   : revenue per unit for record i.          (from column “Revenue”)
    d_i   : demand for record i.                    (from column “Demand”)
    s_i   : initial inventory for record i.         (from column “Initial Inventory”)

Variables
    x_i   ≥ 0   : quantity of record i to sell.     (continuous)

Objective
    maximize   Σ_{i∈I}  r_i · x_i

Constraints
    0 ≤ x_i ≤ d_i   ∀ i∈I          (cannot sell more than demand)
    0 ≤ x_i ≤ s_i   ∀ i∈I          (cannot sell more than inventory)
```

---

**Data Mapping Rules**

| Source | Table ID          | Column in source | Parameter in model |
|--------|-------------------|------------------|--------------------|
| 1      | `file_0_view_0`   | `Revenue`        | `r_i`              |
| 1      | `file_0_view_0`   | `Demand`         | `d_i`              |
| 1      | `file_0_view_0`   | `Initial Inventory` | `s_i`          |
| 1      | `file_0_view_0`   | `source_row`     | index `i`          |

*Each record returned in the observation (8 rows) becomes an element of `I` identified by its `source_row`.  The parameters `r_i`, `d_i`, and `s_i` are taken directly from the corresponding columns of that record.*