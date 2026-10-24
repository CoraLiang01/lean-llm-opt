**Mathematical Model**

Let  

* \(I\) – set of all Fashion products (identified by *Product Name*).  
* \(p_i\) – revenue per unit of product \(i\).  
* \(d_i\) – deterministic demand for product \(i\).  
* \(s_i\) – initial inventory available for product \(i\).  
* \(x_i\) – number of units of product \(i\) to be sold (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\le\; s_i &&\forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column |
|----------------------|--------------|--------|
| \(I\) (index set)   | `file_0_view_0` | `Product Name` |
| \(p_i\)              | `file_0_view_0` | `Revenue` |
| \(d_i\)              | `file_0_view_0` | `Demand` |
| \(s_i\)              | `file_0_view_0` | `Initial Inventory` |
| \(x_i\)              | – (decision variable) | – |

All numeric columns are interpreted as real numbers; the decision variables are constrained to be non‑negative integers. No other constraints are required beyond the demand and inventory limits.