**Mathematical Model**

Let  

- \(I=\{S1,S2,S3,S4,S5\}\) (warehouses)  
- \(J=\{D1,D2,D3,D4,D5\}\) (stores)

Decision variables  
\[
x_{ij}\;\ge 0 \qquad \forall i\in I,\; j\in J
\]

Parameters  

\[
\begin{aligned}
d_j &\;=\;\text{daily demand of store }j,\\
s_i &\;=\;\text{daily supply capacity of warehouse }i,\\
c_{ij} &\;=\;\text{transportation cost per unit from }i\text{ to }j .
\end{aligned}
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

\[
\begin{aligned}
\text{(Demand)}\quad &\sum_{i\in I} x_{ij} \;\ge\; d_j, &&\forall j\in J,\\[4pt]
\text{(Supply)}\quad &\sum_{j\in J} x_{ij} \;\le\; s_i, &&\forall i\in I,\\[4pt]
\text{(Non‑negativity)}\quad &x_{ij}\;\ge\;0, &&\forall i\in I,\; j\in J.
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(I\) | `file_1_view_0` | `region` | All distinct values |
| \(J\) | `file_0_view_0` | `customer` | All distinct values |
| \(d_j\) | `file_0_view_0` | `demand` | Row indexed by `customer` |
| \(s_i\) | `file_1_view_0` | `supply_capacity` | Row indexed by `region` |
| \(c_{ij}\) | `file_2_view_0` | `D1`–`D5` | Row indexed by `Unnamed: 0` (region), column by `D1`–`D5` (customer) |

All indices and parameters are taken directly from the returned rows of the respective tables; no aggregation or transformation is applied beyond the mapping shown.