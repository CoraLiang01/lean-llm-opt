**Mathematical Model**

Let  

- \(I\) be the set of Fashion products (indexed by the *Product Name*).  
- \(Revenue_i\) – revenue per unit of product \(i\).  
- \(Demand_i\) – deterministic demand for product \(i\).  
- \(Inventory_i\) – initial inventory of product \(i\).  
- \(x_i\) – number of units of product \(i\) to be sold.

\[
\begin{aligned}
\max\;& \sum_{i\in I} Revenue_i \, x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le Demand_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le Inventory_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Index | Source Table | Column |
|-------------------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(Revenue_i\) | `file_0_view_0` | `Revenue` |
| \(Demand_i\) | `file_0_view_0` | `Demand` |
| \(Inventory_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`; no additional data are introduced.