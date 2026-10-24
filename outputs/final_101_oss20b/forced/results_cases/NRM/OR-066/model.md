**Symbolic Model**

Let  

* \(I\) – set of suppliers (indexed by \(i\)).  
* \(J\) – set of supermarkets (indexed by \(j\)).  

Parameters  

* \(f_i\) – fixed activation cost of supplier \(i\).  
* \(c_{ij}\) – per‑unit transportation cost from supplier \(i\) to supermarket \(j\).  
* \(d_j\) – total demand of supermarket \(j\).  
* \(M_j = d_j\) – a convenient big‑\(M\) for linking constraints.  

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is activated, 0 otherwise.  
* \(x_{ij} \ge 0\) – quantity supplied from \(i\) to \(j\).  

\[
\begin{aligned}
\min\;& \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij} \;=\; d_j , \qquad \forall j\in J \quad\text{(demand satisfaction)}\\[4pt]
& x_{ij} \;\le\; M_j\,y_i , \qquad \forall i\in I,\; j\in J \quad\text{(linking)}\\[4pt]
& y_i \in \{0,1\}, \qquad \forall i\in I \\[2pt]
& x_{ij} \ge 0 , \qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(c_{ij}\) | `file_2_view_0` | `C1`, `C2`, … (columns for each supermarket) |
| \(d_j\) | `file_0_view_0` | `demand` |

*Index sets*  
* \(I\) is derived from the rows of `file_1_view_0` (suppliers `S1`, `S2`, …).  
* \(J\) is derived from the columns of `file_2_view_0` excluding the supplier identifier column (`Unnamed: 0`).  

No numerical values are embedded in the model; all coefficients are linked to the specified source tables and columns.