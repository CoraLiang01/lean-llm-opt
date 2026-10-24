**Abstract Optimization Model**

```ampl
# Sets
set I;                     # Products (Product_Reference)

# Parameters
param rev {I};             # Revenue per unit (Revenue)
param dem {I};             # Demand quantity (Demand)
param inv {I};             # Initial inventory (Initial Inventory)

# Decision Variables
var x {I} >= 0;            # Units of product i to fulfill

# Objective
maximize TotalRevenue:
    sum {i in I} rev[i] * x[i];

# Constraints
subject to
    DemandLimit {i in I} : x[i] <= dem[i];
    InventoryLimit {i in I} : x[i] <= inv[i];
```

---

**Data Mapping Rules**

| Source | Table ID          | Column Name          | Mapping Target |
|--------|-------------------|----------------------|----------------|
| Index  | `file_0_view_0`   | `Product_Reference`  | `I`            |
| Param  | `file_0_view_0`   | `Revenue`            | `rev`          |
| Param  | `file_0_view_0`   | `Demand`             | `dem`          |
| Param  | `file_0_view_0`   | `Initial Inventory`  | `inv`          |

*Each row in `file_0_view_0` corresponds to a unique product `i ∈ I`.  
The parameters `rev[i]`, `dem[i]`, and `inv[i]` are taken directly from the
respective columns of that row.*