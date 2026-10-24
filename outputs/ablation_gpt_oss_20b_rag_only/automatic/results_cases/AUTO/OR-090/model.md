**Symbolic Model**

```
SETS
    P   : products          {product}
    R   : resources         {resource}

PARAMETERS
    profit_per_unit[p]          : profit earned per unit of product p
    r1_per_unit[p]              : R1 consumption per unit of product p
    r2_per_unit[p]              : R2 consumption per unit of product p
    r3_per_unit[p]              : R3 consumption per unit of product p
    upper_demand_units[p]       : maximum market demand (units) for product p
    capacity[r]                 : total available amount of resource r
    batch_size_units            : fixed batch size in units (10)

VARIABLES
    x[p]  : number of batches produced of product p   (integer, ≥0)

OBJECTIVE
    maximize  Z = Σ_{p∈P}  batch_size_units * x[p] * profit_per_unit[p]

CONSTRAINTS
    /* Resource limits – one constraint per resource */
    r1_cons: Σ_{p∈P} batch_size_units * x[p] * r1_per_unit[p]  ≤  capacity['R1']
    r2_cons: Σ_{p∈P} batch_size_units * x[p] * r2_per_unit[p]  ≤  capacity['R2']
    r3_cons: Σ_{p∈P} batch_size_units * x[p] * r3_per_unit[p]  ≤  capacity['R3']

    /* Demand limits – one per product */
    demand[p]:  batch_size_units * x[p]  ≤  upper_demand_units[p]   ∀p∈P

    /* Non‑negativity and integrality */
    x[p]  ≥ 0   ∀p∈P
    x[p]  ∈ ℤ   ∀p∈P
```

---

**Data Mapping (CSVQA_DATA)**  

| Parameter / Index | Table ID          | Column Name          |
|-------------------|-------------------|----------------------|
| `P` (product)     | `file_0_view_0`   | `product`            |
| `R` (resource)    | `file_1_view_0`   | `resource`           |
| `profit_per_unit` | `file_0_view_0`   | `profit_per_unit`    |
| `r1_per_unit`     | `file_0_view_0`   | `r1_per_unit`        |
| `r2_per_unit`     | `file_0_view_0`   | `r2_per_unit`        |
| `r3_per_unit`     | `file_0_view_0`   | `r3_per_unit`        |
| `upper_demand_units` | `file_0_view_0` | `upper_demand_units` |
| `capacity`        | `file_1_view_0`   | `capacity`           |
| `batch_size_units`| `file_0_view_0`   | `batch_size_units`   |

All parameters are taken directly from the corresponding CSV tables; no additional data are invented. The model respects the batch‑size, resource, and demand constraints exactly as specified in the query.