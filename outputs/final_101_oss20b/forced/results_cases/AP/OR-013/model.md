**Mathematical Model**

Let  

- \(I\) – set of “4U” products (indexed by product name).  
- \(r_i\) – revenue per unit of product \(i\).  
- \(d_i\) – known demand of product \(i\).  
- \(inv_i\) – initial inventory of product \(i\).  
- \(x_i\) – number of units of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad
& 0 \le x_i \le d_i && \forall i\in I \quad\text{(demand limit)}\\
& 0 \le x_i \le inv_i && \forall i\in I \quad\text{(inventory limit)}\\
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(inv_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Product identifiers | `file_0_view_0` | `Product Name` |

All values are taken directly from the filtered rows of the CSV file `OnlineSalesinUSA.csv` where the product name contains “4U”. No additional numerical values or constraints are introduced beyond those specified.