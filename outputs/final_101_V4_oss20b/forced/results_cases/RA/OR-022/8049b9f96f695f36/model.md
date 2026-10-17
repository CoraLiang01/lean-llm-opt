**Mathematical Model**

Let  

* \(I\) be the set of 27‑inch products (indexed by the product name).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand (from column **Demand**)  
  * \(s_i\) – initial inventory (from column **Initial Inventory**)  
  * \(x_i\) – number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i && \forall i\in I \\[2pt]
             & x_i \;\le\; s_i && \forall i\in I \\[2pt]
             & x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | `file_0_view_0` | `Product Name` |

All values are taken directly from the returned rows of the filtered view `file_0_view_0`. No additional data or assumptions are introduced.