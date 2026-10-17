**Mathematical Model**

Let  

- \(P\) be the set of product names extracted from the table.  
- \(r_p\) – revenue per unit of product \(p\).  
- \(d_p\) – demand for product \(p\).  
- \(i_p\) – initial inventory of product \(p\).  
- \(x_p\) – number of units of product \(p\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{p\in P} r_p\,x_p \\[4pt]
\text{s.t. } & 0 \le x_p \le d_p && \forall p\in P \\[2pt]
             & 0 \le x_p \le i_p && \forall p\in P \\[2pt]
             & x_p \in \mathbb{Z}_{\ge 0} && \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(P\) | Product names | `file_0_view_0` | `Product Name` |
| \(r_p\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_p\) | Demand | `file_0_view_0` | `Demand` |
| \(i_p\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the single source table `file_0_view_0`. No additional constraints or parameters are introduced beyond those specified.