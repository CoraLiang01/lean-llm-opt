**Mathematical Model**

```
SETS
    I   :  product identifiers (ProductName)

PARAMETERS
    Value_i   :  benefit coefficient of product i
    Weight_i  :  weight per unit of product i
    Capacity  :  overall inventory capacity

VARIABLES
    x_i ∈ ℤ_+   for all i ∈ I   (daily order quantity of product i)

OBJECTIVE
    maximize  Σ_{i∈I}  Value_i * x_i

CONSTRAINTS
    TotalWeight :  Σ_{i∈I}  Weight_i * x_i  ≤  Capacity
```

---

**Data Mapping (CSVQA_DATA)**  

| Parameter | Source Table | Column | Key (if any) |
|-----------|--------------|--------|--------------|
| `Value_i` | `file_1_view_0` | `Value` | `ProductName` = i |
| `Weight_i`| `file_1_view_0` | `Weight`| `ProductName` = i |
| `Capacity`| `file_0_view_0` | `Capacity`| (single row, no key) |

*All other data (e.g., `ProductName` list) are used only to define the index set `I`.*