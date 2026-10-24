**Mathematical model (facility‑location formulation)**  

```
Sets
    I   : suppliers
    J   : customers

Parameters
    demand_j          : demand of customer j
    fixed_cost_i      : fixed opening cost of supplier i
    transport_cost_ij : unit transportation cost from supplier i to customer j

Decision variables
    y_i   ∈ {0,1}          : 1 if supplier i is opened, 0 otherwise
    x_ij  ≥ 0              : quantity shipped from supplier i to customer j

Objective
    minimize  Σ_{i∈I}  fixed_cost_i · y_i
             + Σ_{i∈I} Σ_{j∈J} transport_cost_ij · x_ij

Constraints
    1. Demand satisfaction
          Σ_{i∈I} x_ij  =  demand_j          ∀ j∈J

    2. Linking (no shipment if supplier closed)
          x_ij  ≤  demand_j · y_i            ∀ i∈I, j∈J
```

---

**Data‑mapping rules**

| Parameter          | Source table | Row key column | Column key / value column |
|--------------------|--------------|----------------|---------------------------|
| `demand_j`         | `file_0_view_0` | `Customer`     | `demand`                 |
| `fixed_cost_i`     | `file_1_view_0` | `Unnamed: 0`   | `fixed_costs`            |
| `transport_cost_ij`| `file_2_view_0` | `Unnamed: 0`   | **customer‑name column** |

*For `transport_cost_ij` the value is taken from the cell whose row matches supplier `i` (via `Unnamed: 0`) and whose column matches customer `j`.  If a customer name in `J` does not appear as a column in `file_2_view_0`, the corresponding `transport_cost_ij` is undefined (the model will be infeasible unless the data are consistent).*

All sets `I` and `J` are taken from the full list of suppliers and customers returned in the observation. No additional limits or capacities are imposed beyond those stated.