**Mathematical Model**

Let  

- \(I=\{i\}\) be the set of suppliers (production plants).  
- \(J=\{j\}\) be the set of customers (retail outlets).  

Parameters  

\[
\begin{aligned}
d_j &\ge 0 &&\text{(daily demand of customer }j)\\
s_i &\ge 0 &&\text{(daily supply capacity of supplier }i)\\
c_{ij} &\ge 0 &&\text{(transportation cost per unit from }i\text{ to }j)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

\[
\begin{aligned}
\text{(Demand)}\quad &\sum_{i\in I} x_{ij} \;\ge\; d_j &&\forall j\in J\\[4pt]
\text{(Supply)}\quad &\sum_{j\in J} x_{ij} \;\le\; s_i &&\forall i\in I\\[4pt]
\text{(Non‑negativity)}\quad &x_{ij}\;\ge\;0 &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(d_j\)   | `file_0_view_0` | `customer` | `demand` |
| \(s_i\)   | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C4` (columns) |

Index sets  

- \(I\) is the set of distinct values in column `Unnamed: 0` of `file_1_view_0`.  
- \(J\) is the set of distinct values in column `customer` of `file_0_view_0`.  

All parameter values are taken directly from the corresponding cells in the tables; no aggregation or transformation is applied beyond the mapping shown.