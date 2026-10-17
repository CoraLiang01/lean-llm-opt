**Mathematical Model**

Let  

- \(I\) – set of tablet models extracted from the filtered table.  
- \(r_i\) – revenue per unit of model \(i\).  
- \(d_i\) – demand quantity of model \(i\).  
- \(inv_i\) – initial inventory of model \(i\).  
- \(x_i\) – number of units of model \(i\) to sell.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le \min\{d_i,\;inv_i\} \qquad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{+} \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue |
| \(d_i\) | Demand | file_0_view_0 | Demand |
| \(inv_i\) | Initial Inventory | file_0_view_0 | Initial Inventory |
| \(I\) | Product Name | file_0_view_0 | Product Name (filtered where contains “TABLET”) |

All parameters are taken directly from the rows returned by the filter; no numerical values are enumerated in the model.