**Mathematical Model**

```math
\begin{aligned}
\text{Sets} \quad & I \;\; \text{: set of Baby products (indexed by product name)} \\[4pt]

\text{Parameters} \quad &
\begin{cases}
\text{Revenue}_i \;\; &\text{revenue per unit of product } i,\\
\text{Demand}_i   \;\; &\text{total demand for product } i,\\
\text{InitialInventory}_i &\text{initial inventory of product } i.
\end{cases} \\[4pt]

\text{Decision Variables} \quad &
x_i \;\; \in \mathbb{Z}_+ \quad \forall i\in I \\[4pt]

\text{Objective} \quad &
\max \;\; \sum_{i\in I} \text{Revenue}_i \, x_i \\[4pt]

\text{Constraints} \quad &
\begin{cases}
x_i \;\le\; \text{Demand}_i          &\forall i\in I,\\
x_i \;\le\; \text{InitialInventory}_i &\forall i\in I.
\end{cases}
\end{aligned}
```

---

**Data Mapping Rules**

| Table ID | Column in source | Parameter / Index mapped to |
|----------|------------------|-----------------------------|
| `file_0_view_0` | `Product Name` | Index set `I` (each distinct product name) |
| `file_0_view_0` | `Revenue` | `Revenue_i` |
| `file_0_view_0` | `Demand` | `Demand_i` |
| `file_0_view_0` | `Initial Inventory` | `InitialInventory_i` |

*For each record in `file_0_view_0`, the value in `Product Name` identifies the index `i`; the corresponding values in the other columns populate the parameters `Revenue_i`, `Demand_i`, and `InitialInventory_i`.*