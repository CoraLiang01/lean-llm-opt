**Symbolic Model**

```math
\begin{aligned}
\text{Sets} \quad & S \;=\;\{S_1,\dots ,S_8\} \\
                   & C \;=\;\{C_1,\dots ,C_9\} \\[4pt]

\text{Parameters} \quad & f_i \;\; (i\in S) \quad\text{: fixed cost of opening supplier }i\\
                         & d_j \;\; (j\in C) \quad\text{: demand of dealership }j\\
                         & t_{ij}\;\; (i\in S,\;j\in C) \quad\text{: transportation cost per vehicle from }i\text{ to }j\\
                         & M \;\;=\;\sum_{j\in C} d_j \quad\text{: big‑M upper bound} \\[4pt]

\text{Decision Variables} \quad & y_i \;\in\{0,1\}\quad (i\in S) \\
                                 & x_{ij}\;\ge 0 \quad (i\in S,\;j\in C) \\[4pt]

\text{Objective} \quad & \min \;\;\sum_{i\in S} f_i\,y_i \;+\; \sum_{i\in S}\sum_{j\in C} t_{ij}\,x_{ij} \\[4pt]

\text{Constraints} \quad & \sum_{i\in S} x_{ij} \;=\; d_j \qquad \forall j\in C \quad\text{(demand satisfaction)}\\
                         & x_{ij} \;\le\; M\,y_i \qquad \forall i\in S,\;j\in C \quad\text{(supplier activation)}\\
                         & y_i \in\{0,1\} \qquad \forall i\in S\\
                         & x_{ij}\;\ge 0 \qquad \forall i\in S,\;j\in C
\end{aligned}
```

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` (row identified by `Unnamed: 0` = supplier id) |
| \(d_j\) | `file_0_view_0` | `demand` (row identified by `customer` = dealership id) |
| \(t_{ij}\) | `file_2_view_0` | `C1`–`C9` (columns) with row identified by `Unnamed: 0` = supplier id |
| \(M\) | computed | \(\sum_{j\in C} d_j\) |

All indices \(i\) and \(j\) are taken from the supplier and dealership identifiers present in the respective tables. No numerical values are hard‑coded in the model; they are supplied through the data mapping above.