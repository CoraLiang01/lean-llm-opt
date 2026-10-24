**Mathematical Model**

```math
\begin{aligned}
\text{Sets} \quad & I \;=\; \{\,i\mid i \text{ is a warehouse}\,\},\\
                  & J \;=\; \{\,j\mid j \text{ is a customer}\,\}.\\[4pt]

\text{Parameters} \quad & \text{fixed\_cost}_i \;\ge 0, \qquad i\in I,\\
                         & \text{demand}_j \;\ge 0, \qquad j\in J,\\
                         & \text{transport\_cost}_{ij} \;\ge 0, \qquad i\in I,\;j\in J.\\[4pt]

\text{Decision Variables} \quad & y_i \in \{0,1\}, \qquad i\in I,\\
                                 & x_{ij} \;\ge 0, \qquad i\in I,\;j\in J.\\[4pt]

\text{Objective} \quad & \min \;\sum_{i\in I}\text{fixed\_cost}_i\,y_i
                        \;+\;\sum_{i\in I}\sum_{j\in J}\text{transport\_cost}_{ij}\,x_{ij}.\\[4pt]

\text{Constraints} \quad
& \sum_{i\in I} x_{ij} \;\ge\; \text{demand}_j,
   \qquad \forall\, j\in J \quad\text{(demand satisfaction)}\\[2pt]
& x_{ij} \;\le\; M\,y_i,
   \qquad \forall\, i\in I,\;j\in J \quad\text{(linking)}\\[2pt]
& y_i \in \{0,1\}, \qquad \forall\, i\in I,\\
& x_{ij} \;\ge\; 0, \qquad \forall\, i\in I,\;j\in J,
\end{aligned}
```

*Here \(M\) is a sufficiently large constant (e.g. \(M=\sum_{j\in J}\text{demand}_j\)).*

---

**Data Mapping**

| Parameter | Source Table | Row‑ID Column | Value Column | Notes |
|-----------|--------------|---------------|--------------|-------|
| `fixed_cost_i` | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Warehouse identifier in `Unnamed: 0` |
| `demand_j` | `file_0_view_0` | `customer` | `demand` | Customer identifier in `customer` |
| `transport_cost_{ij}` | `file_2_view_0` | `Unnamed: 0` | `C1`–`C7` | Row `Unnamed: 0` gives warehouse `i`; columns `C1`–`C7` give costs to customers `j` |

All mappings use the exact table IDs and column names as listed above. No additional data or assumptions are introduced beyond what is specified in the query and the provided CSV files.