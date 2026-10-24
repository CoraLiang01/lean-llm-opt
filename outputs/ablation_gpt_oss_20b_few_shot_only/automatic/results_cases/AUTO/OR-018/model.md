**Symbolic Model**

```
SETS
    I   : set of Baby products

PARAMETERS
    r[i]   : revenue per unit of product i          (column “Revenue”)
    d[i]   : demand for product i                   (column “Demand”)
    I[i]   : initial inventory of product i        (column “Initial Inventory”)

VARIABLES
    x[i]   : units of product i to fulfill          (integer, x[i] ≥ 1)

OBJ
    maximize  Σ{i∈I}  r[i] * x[i]

CONSTRAINTS
    1.  x[i] ≤ d[i]          ∀ i∈I          (cannot exceed demand)
    2.  x[i] ≤ I[i]          ∀ i∈I          (cannot exceed inventory)
```

**Data Mapping**

| Parameter | Source File | Source Column | Mapping Rule |
|-----------|-------------|---------------|--------------|
| `r[i]` | `Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv` | `Revenue` | `r[i]` takes the value in the *Revenue* column for product *i*. |
| `d[i]` | same file | `Demand` | `d[i]` takes the value in the *Demand* column for product *i*. |
| `I[i]` | same file | `Initial Inventory` | `I[i]` takes the value in the *Initial Inventory* column for product *i*. |
| `I` | same file | `Product Name` | `I` is the subset of rows whose *Product Name* begins with the string “Baby”. |

All parameters are bound to the exact columns of the specified CSV file; no numerical values are listed in the model. The decision variables are integer and bounded by the demand and inventory limits, and the objective is to maximize total revenue from the Baby product line.