**Mathematical Model**

```
SETS
    I   : products  (Full_Product_Name)

PARAMETERS
    Revenue[I]          : revenue per unit  (column  Revenue)
    Demand[I]           : demand quantity   (column  Demand)
    InitialInv[I]       : initial inventory (column  Initial Inventory)

VARIABLES
    x[I]  >= 0          : units of product i to fulfill

OBJ
    maximize  Z = sum{i in I} Revenue[i] * x[i]

CONSTRAINTS
    1.  x[i] <= Demand[i]          for all i in I
    2.  x[i] <= InitialInv[i]      for all i in I
```

**Data Mapping**

| Parameter / Set | Table ID          | Column Name          |
|-----------------|-------------------|----------------------|
| I               | file_0_view_0     | Full_Product_Name    |
| Revenue[i]      | file_0_view_0     | Revenue              |
| Demand[i]       | file_0_view_0     | Demand               |
| InitialInv[i]   | file_0_view_0     | Initial Inventory    |

*All indices, parameters, and variables are defined exactly as above; no additional data or constraints are introduced.*