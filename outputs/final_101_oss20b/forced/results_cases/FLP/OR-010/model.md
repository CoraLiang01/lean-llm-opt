**Mathematical Model**

Let  

- \(I\) be the set of products (identified by *Product Name*).  
- Parameters  
  - \(r_i\) – revenue per unit of product \(i\) (from column **Revenue**).  
  - \(d_i\) – demand for product \(i\) (from column **Demand**).  
  - \(s_i\) – initial inventory of product \(i\) (from column **Initial Inventory**).  
- Decision variable  
  - \(x_i \in \mathbb{Z}_{+}\) – quantity of product \(i\) that is fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i && \forall i\in I \\[2pt]
             & x_i \;\le\; s_i && \forall i\in I \\[2pt]
             & x_i \;\in\; \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Product identifiers | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are read directly from the corresponding columns of the table `file_0_view_0`. No additional constraints or parameters are introduced beyond those specified.