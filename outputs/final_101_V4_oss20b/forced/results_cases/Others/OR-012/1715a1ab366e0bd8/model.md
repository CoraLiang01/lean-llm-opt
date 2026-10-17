**Mathematical Model**

Let  

* \(P\) – set of all products (119 items).  
* For each \(p\in P\):  
  * \(r_p\) – revenue per unit (R\$).  
  * \(d_p\) – known demand (units).  
  * \(I_p\) – initial inventory (units).  

Decision variable  

* \(x_p\) – number of units of product \(p\) to sell (integer, \(x_p\ge 0\)).

\[
\begin{aligned}
\max\;& \sum_{p\in P} r_p\,x_p \\[4pt]
\text{s.t.}\;& 0 \le x_p \le d_p && \forall p\in P \\[2pt]
& 0 \le x_p \le I_p && \forall p\in P \\[2pt]
& x_p \in \mathbb{Z} && \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Parameter / Variable | Description |
|----------|--------|----------------------|-------------|
| file_0_view_0 | Product Name | index set \(P\) | Each distinct product name |
| file_0_view_0 | Revenue | \(r_p\) | Revenue per unit for product \(p\) |
| file_0_view_0 | Demand | \(d_p\) | Known demand for product \(p\) |
| file_0_view_0 | Initial Inventory | \(I_p\) | Initial stock of product \(p\) |
| file_0_view_0 | – | \(x_p\) | Decision variable: units sold of product \(p\) |

All parameters are taken directly from the corresponding columns of the CSV file; no additional processing or aggregation is performed. The model uses the full set of products returned by the observation.