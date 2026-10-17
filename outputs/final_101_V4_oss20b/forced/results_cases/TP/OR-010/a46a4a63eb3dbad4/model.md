**Mathematical Model**

Let  

- \(I\) : set of products (indexed by product name).  
- \(r_i\) : revenue per unit of product \(i\).  
- \(d_i\) : deterministic demand of product \(i\).  
- \(s_i\) : initial inventory of product \(i\).  
- \(x_i\) : integer quantity of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; s_i &&\forall i\in I \\[2pt]
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
| \(x_i\) | Decision variable | – | Integer, non‑negative, defined for each \(i\in I\). |

All parameters are taken directly from the rows of `file_0_view_0`; no aggregation or additional processing is performed.