**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & 
\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} x_{ij}=d_j , &&\forall j\in J \qquad\text{(demand)}\\[4pt]
& x_{ij}\;\le\; d_j\,y_i , &&\forall i\in I,\; j\in J \qquad\text{(supplier activation)}\\[4pt]
& y_i\in\{0,1\}, &&\forall i\in I\\
& x_{ij}\ge 0 , &&\forall i\in I,\; j\in J
\end{aligned}
\]

*Sets*  
\(I=\{S1,S2\}\) – suppliers  
\(J=\{C1,C2\}\) – customers  

*Parameters*  
\(d_j\) – demand of customer \(j\)  
\(f_i\) – fixed activation cost of supplier \(i\)  
\(c_{ij}\) – per‑unit transportation cost from supplier \(i\) to customer \(j\)

---

**Data Mapping**

| Parameter | Table ID | Key Column | Value Column | Notes |
|-----------|----------|------------|--------------|-------|
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | Demand for each customer |
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Fixed cost for each supplier |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` (row) | `C1`, `C2` (columns) | Transportation cost matrix |

All indices and parameters are bound directly to the corresponding rows and columns of the supplied CSV files; no additional data are introduced.