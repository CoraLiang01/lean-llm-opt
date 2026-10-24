**Mathematical Model**

Let  

* \(I\) – set of all products (identified by *Product Name*).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – known demand (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) to sell (continuous).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
& 0 \le x_i \le s_i &&\forall i\in I
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Table ID          | Column in Model | Source Column |
|-------------------|-----------------|---------------|
| `file_0_view_0`   | `i` (index)     | `Product Name`|
| `file_0_view_0`   | `r_i`           | `Revenue`     |
| `file_0_view_0`   | `d_i`           | `Demand`      |
| `file_0_view_0`   | `s_i`           | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or aggregation is introduced.