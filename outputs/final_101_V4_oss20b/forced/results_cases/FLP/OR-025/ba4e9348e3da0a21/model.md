**Mathematical Model**

Let  

* \(I\) – set of tablet models (indexed by the product name).  
* \(r_i\) – revenue per unit of model \(i\).  
* \(d_i\) – deterministic demand for model \(i\).  
* \(inv_i\) – initial inventory of model \(i\).  
* \(x_i\) – number of units of model \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
& x_i \ge 0 && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(I\) | Product identifiers | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(inv_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All numerical values are taken directly from the rows returned by the filter on `Product Name` containing `"TABLET"`. No additional constraints or parameters are introduced beyond those specified.