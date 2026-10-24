**Mathematical Model**

Let  

* \(I=\{i\}\) be the set of real‑estate areas (ProductName).  
* \(x_i\) – scale of development per day in area \(i\) (continuous, \(x_i\ge 0\)).  

Parameters  

* \(b_i\) – benefit per unit of development in area \(i\).  
* \(w_i\) – weight (resource consumption) per unit of development in area \(i\).  
* \(C\) – overall development capacity limit.

\[
\begin{aligned}
\max\;& \sum_{i\in I} b_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Parameter | Source |
|----------|--------|-----------|--------|
| `file_0_view_0` | `Capacity` | \(C\) | capacity.csv |
| `file_1_view_0` | `ProductName` | index set \(I\) | products.csv |
| `file_1_view_0` | `Value` | \(b_i\) | products.csv |
| `file_1_view_0` | `Weight` | \(w_i\) | products.csv |

All parameters are taken directly from the corresponding columns; no additional processing or aggregation is performed.