**Mathematical Model**

Let  

* \(I\) be the set of products whose *Product_Reference* starts with “ELE‑S” (12 items).  
* For each \(i\in I\)  
  * \(R_i\) – revenue per unit (RMB)  
  * \(D_i\) – deterministic demand (units)  
  * \(S_i\) – initial inventory (units)  

Decision variable  

* \(x_i\in\mathbb{Z}_{\ge 0}\) – units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} R_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le D_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le S_i &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Source |
|----------|--------|--------|
| file_0_view_0 | Product_Reference | `SalesStoreoverview.csv` (filtered: prefix “ELE‑S”) |
| file_0_view_0 | Revenue | `SalesStoreoverview.csv` |
| file_0_view_0 | Demand | `SalesStoreoverview.csv` |
| file_0_view_0 | Initial Inventory | `SalesStoreoverview.csv` |

All parameters \(R_i, D_i, S_i\) are taken directly from the corresponding columns of the filtered rows. The index set \(I\) consists of the 12 product references returned by the filter. No additional rows or columns are introduced.