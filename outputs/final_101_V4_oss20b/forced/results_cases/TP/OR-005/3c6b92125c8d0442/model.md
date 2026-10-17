**Mathematical model**

```
Sets
    I = {supplier1, supplier2, supplier3, supplier4,
         supplier5, supplier6, supplier7, supplier8}
    J = {demand1, demand2, demand3, demand4,
         demand5, demand6, demand7, demand8}

Parameters
    d_j   : demand of customer j          (from file_0_view_0)
    s_i   : supply capacity of supplier i  (from file_1_view_0)
    c_{i,j} : transportation cost per unit
              defined as
                if j ∈ {demand1,demand2,demand3,demand4} then
                    value from file_2_view_0 where Unnamed:0 = i
                else
                    0

Variables
    x_{i,j}  ≥ 0   quantity shipped from supplier i to customer j

Objective
    minimize   Σ_{i∈I} Σ_{j∈J} c_{i,j} · x_{i,j}

Constraints
    Demand   : Σ_{i∈I} x_{i,j}  ≥  d_j      ∀ j∈J
    Supply   : Σ_{j∈J} x_{i,j}  ≤  s_i      ∀ i∈I
```

---

**Data‑mapping rules**

| Source table | Source column | Target set / parameter | Mapping rule |
|--------------|---------------|------------------------|--------------|
| `file_0_view_0` | `Customers` | `J` | Each distinct value becomes an element of `J`. |
| | `demand` | `d_j` | `d_j = demand` for the customer identified by `Customers`. |
| `file_1_view_0` | `Supplier` | `I` | Each distinct value becomes an element of `I`. |
| | `supply_capacity` | `s_i` | `s_i = supply_capacity` for the supplier identified by `Supplier`. |
| `file_2_view_0` | `Unnamed: 0` | `I` | Each distinct value becomes an element of `I`. |
| | `demand1`–`demand4` | `c_{i,j}` | For each row, let `i = Unnamed:0`.  
  For each column `demandk` (k=1..4), set `c_{i,demandk} = value`. |
| | – | `c_{i,j}` for `j ∈ {demand5,demand6,demand7,demand8}` | `c_{i,j} = 0` for all `i ∈ I`. |

These rules bind every parameter in the model to the corresponding source data or a deterministic expression, ensuring the model is fully specified without inventing any missing values.