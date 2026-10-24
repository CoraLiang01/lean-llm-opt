**Mathematical Model**

Let  

* \(I\) – set of Fashion products (identified by *Product Name*).  
* \(p_i\) – revenue per unit of product \(i\).  
* \(d_i\) – deterministic demand for product \(i\).  
* \(s_i\) – initial inventory of product \(i\).  
* \(x_i\) – number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(p_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`; no values are enumerated or altered.