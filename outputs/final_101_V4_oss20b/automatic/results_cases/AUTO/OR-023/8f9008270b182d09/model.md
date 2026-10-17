**Mathematical Model**

Let  

- \(I\) be the set of products with reference starting with “ELE-ACC-”.  
- \(r_i\) be the revenue per unit of product \(i\).  
- \(d_i\) be the deterministic demand for product \(i\).  
- \(s_i\) be the initial inventory of product \(i\).  
- \(x_i\) be the number of units of product \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product_Reference` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – |

All indices and parameters are taken directly from the rows returned by the query; no values are enumerated in the model.