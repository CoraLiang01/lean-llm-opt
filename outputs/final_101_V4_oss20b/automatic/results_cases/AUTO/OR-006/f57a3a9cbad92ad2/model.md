**Mathematical Model**

Let  

* \(I=\{S1,S2,\dots ,S10\}\) (warehouse indices)  
* \(J=\{C1,C2,\dots ,C10\}\) (retail‑store indices)

Parameters  

* \(d_j\) – daily demand of store \(j\)  \(j\in J\)  
* \(s_i\) – daily supply capacity of warehouse \(i\) \(i\in I\)  
* \(c_{ij}\) – transportation cost per unit from warehouse \(i\) to store \(j\) \(i\in I,\;j\in J\)

Decision variables  

* \(x_{ij}\ge 0\) quantity shipped from warehouse \(i\) to store \(j\) \(i\in I,\;j\in J\)

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij} \;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column | Index Mapping |
|-----------|--------------|------------|--------------|---------------|
| \(d_j\)   | `file_0_view_0` | `customer` | `demand` | \(j\) indexed by `customer` |
| \(s_i\)   | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` | \(i\) indexed by `Unnamed: 0` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C10` (columns) | \(i\) indexed by `Unnamed: 0`, \(j\) indexed by column names `C1`…`C10` |

All tables are 10 × 10 as confirmed by the validation matrix check. No additional data or constraints are required.