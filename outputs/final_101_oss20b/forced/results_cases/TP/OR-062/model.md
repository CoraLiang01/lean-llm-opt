**Mathematical Model**

Let  

* \(I\) – set of suppliers (rows of *fixed_cost.csv*).  
* \(J\) – set of licensed stores (rows of *demand.csv*).  

Parameters  

* \(d_j\) – demand of store \(j\) (from *demand.csv*).  
* \(f_i\) – fixed opening cost of supplier \(i\) (from *fixed_cost.csv*).  
* \(c_{ij}\) – unit transportation cost from supplier \(i\) to store \(j\) (from *transportation_costs.csv*).  
* \(M = \sum_{j\in J} d_j\) – a sufficiently large constant.

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise.  
* \(x_{ij} \ge 0\) – quantity shipped from supplier \(i\) to store \(j\).

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} \;\ge\; d_j , && \forall j\in J \quad\text{(demand)}\\[4pt]
& x_{ij} \;\le\; M\,y_i , && \forall i\in I,\; j\in J \quad\text{(activation)}\\[4pt]
& y_i \in \{0,1\}, && \forall i\in I\\
& x_{ij} \;\ge\; 0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(I\) | `file_1_view_0` | `Unnamed: 0` | Supplier names |
| \(J\) | `file_0_view_0` | `Customer` | Store names |
| \(d_j\) | `file_0_view_0` | `demand` | Demand per store |
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Fixed cost per supplier |
| \(c_{ij}\) | `file_2_view_0` | `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` | Row = supplier (`Unnamed: 0`), column = store |
| \(M\) | Computed | – | \(M = \sum_{j\in J} d_j\) |

All indices, parameters, and variables are defined exactly as above; no additional data are introduced.