**Mathematical Model**

Let  

- \(I\) be the set of pizza products.  
- For each \(i\in I\):  
  - \(r_i\) – revenue per unit (from column **Revenue**).  
  - \(d_i\) – demand (from column **Demand**).  
  - \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \quad\text{(demand)}\\
             & x_i \;\le\; s_i &&\forall i\in I \quad\text{(inventory)}\\
             & x_i \;\in\; \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column | Mapping |
|-----------|--------------|--------|---------|
| \(I\) | `file_0_view_0` | `Product Name` | Each distinct product name defines an index \(i\). |
| \(r_i\) | `file_0_view_0` | `Revenue` | `r_i = Revenue` for product \(i\). |
| \(d_i\) | `file_0_view_0` | `Demand` | `d_i = Demand` for product \(i\). |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | `s_i = Initial Inventory` for product \(i\). |

All parameters are taken directly from the corresponding columns of the single source table `file_0_view_0`. No additional data or aggregation is introduced.