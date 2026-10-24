**Symbolic Model**

```
Indices
    i ∈ I          // all produce types

Parameters
    w_i            // weight of one unit of type i
    v_i            // benefit (value) of one unit of type i
    C              // total inventory capacity

Variables
    x_i ∈ ℤ_+      // daily order quantity of type i

Objective
    maximize   ∑_{i∈I} v_i · x_i

Constraints
    1.  ∑_{i∈I} w_i · x_i  ≤  C          // total weight cannot exceed capacity
```

---

**Data Mapping Rules**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| `I`       | `file_1_view_0` | `ProductName` |
| `w_i`     | `file_1_view_0` | `Weight` |
| `v_i`     | `file_1_view_0` | `Value` |
| `C`       | `file_0_view_0` | `Capacity` |

*All mappings use the exact table identifiers and column names provided in the observation.*