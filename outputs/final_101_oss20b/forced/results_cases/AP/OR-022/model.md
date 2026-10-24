**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* contains “27in” (returned rows).  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**)  
  - \(d_i\) – demand quantity (from column **Demand**)  
  - \(inv_i\) – initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i\) – number of units of product \(i\) to be sold (fulfilled).

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le inv_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Returned rows | `file_0_view_0` | – |
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(inv_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – | – |

All parameters are taken directly from the filtered view of **Salesorders.csv** where *Product Name* contains “27in”. No additional numerical values are enumerated.