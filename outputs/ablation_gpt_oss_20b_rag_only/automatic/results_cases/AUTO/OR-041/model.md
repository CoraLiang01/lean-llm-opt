**Model**

```
Index sets
----------
I = { i | i ∈ file_1_view_0.ProductName }

Parameters
----------
Value[i]   = file_1_view_0.Value[i]
Weight[i]  = file_1_view_0.Weight[i]
Capacity   = file_0_view_0.Capacity

Variables
---------
x[i]  ≥ 0          ∀ i ∈ I          (continuous development scale)

Objective
---------
maximize  Z = Σ_{i∈I}  Value[i] * x[i]

Constraint
----------
Σ_{i∈I}  Weight[i] * x[i]  ≤  Capacity
```

**Data Mapping**

| Parameter | table_id        | column |
|-----------|-----------------|--------|
| Value[i]  | file_1_view_0   | Value  |
| Weight[i] | file_1_view_0   | Weight |
| Capacity  | file_0_view_0   | Capacity |

*All indices, parameters, and variables are defined exactly as above; no numeric values are hard‑coded.*