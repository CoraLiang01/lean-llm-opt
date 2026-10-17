**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; M\,y_i , &&\forall i\in I,\; j\in J \quad\text{(link)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J\\[4pt]
&y_i\in\{0,1\}, &&\forall i\in I
\end{aligned}
\]

where  

\[
M \;=\; \sum_{j\in J} d_j
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column(s) |
|-----------|--------------|------------|-----------------|
| \(d_j\)   | `file_0_view_0` | `customer` | `demand` |
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` | `C1`, `C2`, `C3`, `C4`, `C5` |

*Indices*  
\(I = \{S1, S2, S3, S4, S5\}\) (suppliers)  
\(J = \{C1, C2, C3, C4, C5\}\) (customers)

All parameters are taken directly from the specified columns; no additional aggregation or transformation is performed.