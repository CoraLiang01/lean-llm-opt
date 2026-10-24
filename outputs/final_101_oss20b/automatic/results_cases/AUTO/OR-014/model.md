**Mathematical Model**

Let  

- \(I\) be the set of pizza products (indexed by the *Product Name* column).  
- \(r_i\) – revenue per unit of product \(i\) (column **Revenue**).  
- \(d_i\) – demand for product \(i\) (column **Demand**).  
- \(inv_i\) – initial inventory of product \(i\) (column **Initial Inventory**).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i && \forall i\in I \\[2pt]
& x_i \;\le\; inv_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(inv_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the rows of the table `file_0_view_0`; no additional data are introduced.