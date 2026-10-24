**Mathematical model**

Let  

* \(I\) – set of products whose *Product_Reference* starts with **ELE‑S**  
  (obtained from the filtered rows of *SalesStoreoverview.csv*).  

Parameters (indexed by \(i\in I\))  

* \(r_i\) – revenue per unit (column **Revenue**)  
* \(d_i\) – demand (column **Demand**)  
* \(s_i\) – initial inventory (column **Initial Inventory**)  

Decision variable  

* \(x_i \ge 0\) – units of product \(i\) to sell (continuous).  

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; d_i &&\forall i\in I \\[2pt]
& x_i \;\le\; s_i &&\forall i\in I \\[2pt]
& x_i \;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Column name | Mapping rule |
|-----------|--------------|-------------|--------------|
| \(I\) | `file_0_view_0` | `Product_Reference` | Include row if value starts with `"ELE-S"` |
| \(r_i\) | `file_0_view_0` | `Revenue` | `r_i = float(row["Revenue"])` |
| \(d_i\) | `file_0_view_0` | `Demand` | `d_i = int(row["Demand"])` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | `s_i = float(row["Initial Inventory"])` |
| \(x_i\) | – | – | Decision variable, continuous, \(x_i \ge 0\) |

*The table id `file_0_view_0` refers to the filtered view of *SalesStoreoverview.csv* that contains only the 12 rows whose `Product_Reference` begins with `"ELE-S"`.*