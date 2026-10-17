**Mathematical Model**

Let  

- \(I\) – set of all products in the dataset.  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**).  
  - \(d_i\) – total demand over the sales horizon (from column **Demand**).  
  - \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to fulfill.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \le d_i && \forall i\in I \quad\text{(demand limit)}\\
& x_i \le s_i && \forall i\in I \quad\text{(inventory limit)}\\
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column Name |
|-----------|--------|----------|-------------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Product set | `file_0_view_0` | `Product Name` |

No other data or constraints are introduced beyond those specified in the query.