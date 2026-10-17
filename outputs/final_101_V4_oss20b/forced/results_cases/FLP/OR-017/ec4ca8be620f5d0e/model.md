**Mathematical Model**

Let  

* \(I\) – set of all SKUs whose code starts with the prefix **ZZ** (obtained from the filter in the query).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(v_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) that the store plans to sell (continuous).

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le v_i && \forall i\in I \\[2pt]
& 0 \le x_i \le d_i && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Set of SKUs with prefix “ZZ” | `file_0_view_0` | `SKU` (filter: prefix “ZZ”) |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No numerical values are enumerated in the model.