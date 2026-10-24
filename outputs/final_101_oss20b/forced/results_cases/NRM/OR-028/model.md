**Mathematical Model**

Let  

- \(I\) be the set of product identifiers (one element per row of the source table).  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (column **Revenue**)  
  - \(d_i\) – deterministic demand over the horizon (column **Demand**)  
  - \(s_i\) – initial inventory (column **Initial Inventory**)  

Decision variable  

- \(x_i\) – number of units of product \(i\) to fulfill (integer, non‑negative).

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad & x_i \;\le\; d_i, \qquad \forall i\in I \\[2pt]
& x_i \;\le\; s_i, \qquad \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{+}, \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Product identifiers | `file_0_view_0` | `Product Name` |

All parameter values are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or constraints are introduced beyond those specified in the query.