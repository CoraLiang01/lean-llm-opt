**Mathematical Model**

Let  

* \(I\) – set of products whose *Product_Reference* starts with “ELE‑S” (12 items).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (RMB)  (from column **Revenue**)  
  * \(d_i\) – demand quantity (units) (from column **Demand**)  
  * \(s_i\) – initial inventory (units) (from column **Initial Inventory**)  
  * \(x_i\) – number of units of product \(i\) to sell (decision variable)

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le s_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | file_0_view_0 | Revenue |
| \(d_i\) | Demand | file_0_view_0 | Demand |
| \(s_i\) | Initial Inventory | file_0_view_0 | Initial Inventory |
| \(x_i\) | Decision variable | – | – |

*Index set \(I\)* is defined by the filter condition  
`Product_Reference` starts with `"ELE-S"` applied to **SalesStoreoverview.csv**.  
All 12 rows that satisfy this filter constitute the index set.