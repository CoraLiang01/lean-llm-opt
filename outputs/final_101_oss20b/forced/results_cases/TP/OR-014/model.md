**Mathematical Model**

Let  

- \(P\) be the set of product names.  
- \(r_p\) – revenue per unit of product \(p\).  
- \(d_p\) – deterministic demand of product \(p\).  
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

**Data Mapping Rules**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(P\) | `file_0_view_0` | `Product Name` |
| \(r_p\) | `file_0_view_0` | `Revenue` |
| \(d_p\) | `file_0_view_0` | `Demand` |
| \(i_p\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or aggregation is performed.