**Mathematical Model**

Let  

* \(I\) – index set of all products whose *Product Name* contains “FAUX” (as returned by the query).  
* \(p_i\) – revenue per unit of product \(i\).  
* \(I_i\) – initial inventory of product \(i\).  
* \(D_i\) – deterministic demand for product \(i\).  
* \(x_i\) – number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le I_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le D_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(p_i\) | `file_0_view_0` | `Revenue` |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` |
| \(D_i\) | `file_0_view_0` | `Demand` |

All parameters are taken directly from the corresponding columns of the table returned by the query. No additional data are introduced.