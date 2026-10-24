**Mathematical Model**

Let  

* \(I\) – set of products (indexed by the product name).  
* \(r_i\) – revenue per unit of product \(i\).  
* \(d_i\) – demand for product \(i\).  
* \(s_i\) – initial inventory of product \(i\).  
* \(x_i\) – number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\le\; s_i &&\forall i\in I \\[2pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column |
|----------------------|--------------|--------|
| \(I\) (index set)   | `file_0_view_0` | `Full_Product_Name` |
| \(r_i\)              | `file_0_view_0` | `Revenue` |
| \(d_i\)              | `file_0_view_0` | `Demand` |
| \(s_i\)              | `file_0_view_0` | `Initial Inventory` |
| \(x_i\)              | – | – (decision variable, integer, non‑negative) |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`; no values are enumerated in the model.