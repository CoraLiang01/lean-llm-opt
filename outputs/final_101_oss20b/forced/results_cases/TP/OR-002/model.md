**Mathematical model**

Let  

* \(I=\{i\mid i\text{ is a supplier in }{\tt file\_1\_view\_0}\}\)  
* \(J=\{j\mid j\text{ is a customer in }{\tt file\_0\_view\_0}\}\)

Parameters  

\[
\begin{aligned}
s_i &\;=\;\text{supply capacity of supplier }i
      &&\text{from }{\tt file\_1\_view\_0.supply\_capacity}\\[4pt]
d_j &\;=\;\text{daily demand of customer }j
      &&\text{from }{\tt file\_0\_view\_0.demand}\\[4pt]
c_{ij}&\;=\;\text{transportation cost per unit from }i\text{ to }j
      &&\text{from }{\tt file\_2\_view\_0}
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\ge 0 \qquad\forall i\in I,\;j\in J
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

\[
\begin{aligned}
\text{(Demand)}\quad &\sum_{i\in I} x_{ij}\;\ge\; d_j
      &&\forall j\in J\\[4pt]
\text{(Supply)}\quad &\sum_{j\in J} x_{ij}\;\le\; s_i
      &&\forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

| Index | Source table | Source column | Mapping rule |
|-------|--------------|---------------|--------------|
| \(I\) | `file_1_view_0` | `Unnamed: 0` | Set of all supplier identifiers |
| \(J\) | `file_0_view_0` | `customer` | Set of all customer identifiers |
| \(s_i\) | `file_1_view_0` | `supply_capacity` | Value for supplier \(i\) |
| \(d_j\) | `file_0_view_0` | `demand` | Value for customer \(j\) |
| \(c_{ij}\) | `file_2_view_0` | column `j` (e.g., `C1`, `C2`, …) | Value in row where `Unnamed: 0` = \(i\) and column = \(j\) |

All tables are used in full; no aggregation or omission of rows is performed. The model respects the original objective sense, variable domains, and constraints as specified in the query.