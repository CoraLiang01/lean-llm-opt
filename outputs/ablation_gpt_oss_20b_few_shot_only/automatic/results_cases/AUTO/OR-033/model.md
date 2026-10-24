**Symbolic Model**

```
SETS
    I   :  set of products whose name starts with "Baby"

PARAMETERS
    Revenue_i   {i ∈ I}   : revenue per unit of product i
    Demand_i    {i ∈ I}   : total demand for product i
    Inventory_i {i ∈ I}   : initial inventory of product i

VARIABLES
    x_i {i ∈ I}   : number of units of product i to fulfill
                   (integer, x_i ≥ 1)

OBJECTIVE
    maximize  Z = Σ_{i∈I} Revenue_i * x_i

CONSTRAINTS
    1.  Demand limit:      x_i ≤ Demand_i          ∀ i ∈ I
    2.  Inventory limit:   x_i ≤ Inventory_i       ∀ i ∈ I
```

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| `Revenue_i`   | `Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv` | `Revenue` |
| `Demand_i`    | `Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv` | `Demand` |
| `Inventory_i` | `Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv` | `Initial Inventory` |
| Index set `I` | Same file | `Product Name` (filter: names beginning with `"Baby"`) |

*No numerical values are included; the model is fully symbolic and references the source data via the mapping table.*