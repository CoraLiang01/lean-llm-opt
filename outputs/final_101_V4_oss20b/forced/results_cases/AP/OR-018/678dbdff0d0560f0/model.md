**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* starts with “Baby” (the rows returned by the filter).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand (from column **Demand**).  
  * \(v_i\) – initial inventory (from column **Initial Inventory**).  
  * \(x_i\) – number of units of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le \min\{d_i,\;v_i\}\qquad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{+}\qquad \forall i\in I
\end{aligned}
\]

The constraint \(x_i \le \min\{d_i,v_i\}\) is equivalent to the two linear inequalities  
\(x_i \le d_i\) and \(x_i \le v_i\).

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is defined by the rows returned from `file_0_view_0` after applying the filter `Product Name` starts with `"Baby"`. No additional rows are included.