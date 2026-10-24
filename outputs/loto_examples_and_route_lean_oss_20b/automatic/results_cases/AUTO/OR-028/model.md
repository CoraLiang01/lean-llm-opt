**Mathematical Model**

Let  

* \(I\) – set of products (indexed by the product name).  
* \(R_i\) – revenue per unit of product \(i\).  
* \(D_i\) – deterministic demand for product \(i\).  
* \(Inv_i\) – initial inventory available for product \(i\).  
* \(x_i\) – number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le D_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le Inv_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|---------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(R_i\) | `file_0_view_0` | `Revenue` |
| \(D_i\) | `file_0_view_0` | `Demand` |
| \(Inv_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`; no additional data are introduced.