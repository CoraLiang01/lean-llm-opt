**Symbolic Model**

```
Sets
    I   : set of FDK57 car‑model records

Parameters
    Revenue_i   : revenue per unit of model i
    Demand_i    : demand quantity of model i
    Inventory_i : initial inventory of model i

Variables
    x_i   : integer quantity of model i to sell,  x_i ∈ ℤ₊

Objective
    maximize   ∑_{i∈I} Revenue_i · x_i

Constraints
    0 ≤ x_i ≤ Demand_i          ∀ i∈I
    0 ≤ x_i ≤ Inventory_i       ∀ i∈I
```

**Data Mapping**

| Symbol | Source Table | Column | Mapping Rule |
|--------|--------------|--------|--------------|
| I      | `file_0_view_0` | – | `I = { data_row_number of each returned record }` |
| Revenue_i | `file_0_view_0` | `Revenue` | `Revenue_i = value in column Revenue for record i` |
| Demand_i | `file_0_view_0` | `Demand` | `Demand_i = value in column Demand for record i` |
| Inventory_i | `file_0_view_0` | `Initial Inventory` | `Inventory_i = value in column Initial Inventory for record i` |

All parameters are taken directly from the returned rows of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.