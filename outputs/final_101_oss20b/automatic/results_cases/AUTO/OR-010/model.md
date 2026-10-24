**Mathematical Model**

Let  

- \(I\) be the set of products.  
- \(rev_i\) be the revenue per unit of product \(i\).  
- \(dem_i\) be the known demand for product \(i\).  
- \(inv_i\) be the initial inventory of product \(i\).  
- \(x_i\) be the number of units of product \(i\) to fulfill.

\[
\begin{aligned}
\max\;& \sum_{i\in I} rev_i \, x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le dem_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(rev_i\) | file_0_view_0 | Revenue |
| \(dem_i\) | file_0_view_0 | Demand |
| \(inv_i\) | file_0_view_0 | Initial Inventory |
| Index \(i\) | file_0_view_0 | Product Name |

All parameters are extracted directly from the specified columns of the single source table `file_0_view_0`. No additional data or assumptions are introduced.