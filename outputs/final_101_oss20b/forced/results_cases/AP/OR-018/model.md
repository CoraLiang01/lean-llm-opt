**Mathematical Model**

Let  

* \(I\) – set of all products whose *Product Name* starts with the prefix **“Baby”** (obtained from the filtered view `file_0_view_0`).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – deterministic demand (from column **Demand**).  
  * \(v_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i\) – number of units of product \(i\) that the retailer chooses to sell during the horizon.

\[
\begin{aligned}
\max_{x}\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. }& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
            & 0 \le x_i \le v_i && \forall i\in I \\[2pt]
            & x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

The two upper‑bound constraints together enforce \(x_i \le \min\{d_i,v_i\}\); no restocking is allowed, so inventory is a hard limit.

---

**Data Mapping**

| Symbol | Source | Table ID | Column | Filter |
|--------|--------|----------|--------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` | `Product Name` starts with `"Baby"` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` | `Product Name` starts with `"Baby"` |
| \(v_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` | `Product Name` starts with `"Baby"` |

The index set \(I\) is defined by the rows returned after applying the filter on *Product Name* in the view `file_0_view_0`. No additional rows or columns are introduced.