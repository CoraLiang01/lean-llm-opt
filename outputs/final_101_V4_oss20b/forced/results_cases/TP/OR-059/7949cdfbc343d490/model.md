**Mathematical Model**

```math
\begin{aligned}
\text{Sets} \quad & I=\{S1,S2,S3,S4,S5,S6,S7,S8\},\\
                  & J=\{C1,C2,C3,C4,C5,C6,C7,C8,C9\}.\\[4pt]

\text{Parameters} \quad & \text{fixed\_cost}_i \;\;(i\in I),\\
                         & \text{demand}_j \;\;(j\in J),\\
                         & \text{cost}_{ij}\;\;(i\in I,\;j\in J),\\
                         & M_i=\sum_{j\in J}\text{demand}_j \;\;(i\in I).\\[4pt]

\text{Decision Variables} \quad & y_i\in\{0,1\}\;\;(i\in I),\\
                                 & x_{ij}\ge 0\;\;(i\in I,\;j\in J).\\[4pt]

\text{Objective} \quad & \min \;\sum_{i\in I}\text{fixed\_cost}_i\,y_i
                         \;+\;\sum_{i\in I}\sum_{j\in J}\text{cost}_{ij}\,x_{ij}.\\[4pt]

\text{Constraints} \quad
&\sum_{i\in I}x_{ij}\;\ge\;\text{demand}_j
   &&\forall j\in J \quad\text{(demand satisfaction)}\\[2pt]
&x_{ij}\;\le\;M_i\,y_i
   &&\forall i\in I,\;j\in J \quad\text{(linking)}\\[2pt]
&y_i\in\{0,1\}
   &&\forall i\in I\\[2pt]
&x_{ij}\ge 0
   &&\forall i\in I,\;j\in J.
\end{aligned}
```

> **Note**  
> The parameter `cost_{ij}` is defined only for the customer indices that appear
> in the transportation‑cost file (`C1`–`C4`).  For the remaining customers
> (`C5`–`C9`) the cost values are not provided in the data set.

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column(s) | Notes |
|-----------|---------------|------------|-----------------|-------|
| `fixed_cost_i` | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Supplier identifier `S1`–`S8` |
| `demand_j` | `file_0_view_0` | `customer` | `demand` | Dealership identifier `C1`–`C9` |
| `cost_{ij}` | `file_2_view_0` | `Unnamed: 0` | `C1`, `C2`, `C3`, `C4` | Only these four customer columns are present |
| `M_i` | Derived | – | – | Sum of all `demand_j` for each supplier `i` |

All indices and identifiers are taken in the order they appear in the respective
source files.  No additional data are invented or omitted.