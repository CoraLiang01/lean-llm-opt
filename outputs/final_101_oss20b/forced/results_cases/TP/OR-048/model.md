**Mathematical Model**

Let  

- \(I=\{\,i\mid i=\text{StorageID from capacity.csv}\,\}\)  
- \(J=\{\,j\mid j=\text{ProductName from products.csv}\,\}\)

Parameters  

- \(c_i\) – capacity of storage area \(i\) (units of weight)  
- \(v_j\) – value of product \(j\) (monetary units)  
- \(w_j\) – weight of product \(j\) (units of weight)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in storage area \(i\)

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Column |
|-----------|--------------|---------------|--------------|
| \(c_i\)   | file_0_view_0 | Capacity      | StorageID    |
| \(v_j\)   | file_1_view_0 | Value         | ProductName  |
| \(w_j\)   | file_1_view_0 | Weight        | ProductName  |

| Variable | Index Set 1 | Index Set 2 |
|----------|-------------|-------------|
| \(x_{ij}\) | StorageID (file_0_view_0) | ProductName (file_1_view_0) |

All indices, parameters, and variables are defined exactly as they appear in the source tables; no additional data or aggregation is introduced.