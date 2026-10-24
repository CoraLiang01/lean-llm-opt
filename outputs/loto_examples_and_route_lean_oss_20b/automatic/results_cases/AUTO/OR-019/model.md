**Mathematical Model**

Let  

* \(I\) – set of products whose *Product Name* contains “27in”.  
* For each \(i\in I\)  
  * \(p_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to fulfill.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i && \forall i\in I \\[2pt]
             & x_i \;\le\; s_i && \forall i\in I \\[2pt]
             & x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(p_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | `file_0_view_0` | `Product Name` |

All parameters are taken directly from the returned rows of the observation; no additional data are introduced.