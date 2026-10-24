**Mathematical Model**

Let  

* \(I\) – set of all products (rows of *SalesDatainBusinesses.csv*).  
* \(r_i\) – revenue per unit of product \(i\) (column **Revenue**).  
* \(d_i\) – demand for product \(i\) (column **Demand**).  
* \(s_i\) – initial inventory of product \(i\) (column **Initial Inventory**).  
* \(x_i\) – number of units of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& x_i \;\le\; d_i \qquad &&\forall i\in I \\[2pt]
& x_i \;\le\; s_i \qquad &&\forall i\in I \\[2pt]
& x_i \;\ge\; 0 \qquad &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z} \qquad &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Parameter |
|----------|--------|-----------|
| file_0_view_0 | Product Name | – (used only for identification) |
| file_0_view_0 | Revenue | \(r_i\) |
| file_0_view_0 | Demand | \(d_i\) |
| file_0_view_0 | Initial Inventory | \(s_i\) |

All parameters are taken directly from the corresponding columns of the table *SalesDatainBusinesses.csv* (table ID `file_0_view_0`). No additional data or transformations are introduced.