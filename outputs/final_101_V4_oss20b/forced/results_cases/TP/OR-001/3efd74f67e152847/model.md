**Mathematical Model**

Let  

- \(I=\{S1,S2,\dots ,S18\}\) (suppliers)  
- \(J=\{C1,C2,\dots ,C18\}\) (customers)  

Parameters  

- \(d_j\) : demand of customer \(j\)  \(j\in J\)  
- \(s_i\) : supply capacity of supplier \(i\)  \(i\in I\)  
- \(c_{ij}\) : transportation cost per unit from supplier \(i\) to customer \(j\) \(i\in I,\;j\in J\)

Decision variables  

- \(x_{ij}\ge 0\) quantity shipped from supplier \(i\) to customer \(j\).

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&\sum_{j\in J} x_{ij} \;\le\; s_i , &&\forall i\in I \quad\text{(supply)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\;j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Row Identifier | Column Identifier |
|-----------|--------------|----------------|-------------------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` |
| \(s_i\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C18` (columns for each customer) |

All identifiers are taken verbatim from the source tables; no aggregation or transformation is applied.