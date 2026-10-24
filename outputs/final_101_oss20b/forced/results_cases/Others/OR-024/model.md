**Mathematical Model**

Let  

* \(I\) – set of products whose *Product Name* starts with the prefix **S700_**  
  (obtained from the filter in the data view).  

For each \(i\in I\) define the parameters  

* \(r_i\) – revenue per unit (from column **Revenue**)  
* \(d_i\) – demand quantity (from column **Demand**)  
* \(s_i\) – initial inventory (from column **Initial Inventory**)  

Decision variable  

* \(x_i\) – number of units of product \(i\) to sell (integer, \(x_i\ge 0\))  

The optimization problem is  

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Columns Used | Filter Condition |
|----------|--------------|------------------|
| `file_0_view_0` | `Product Name`, `Revenue`, `Demand`, `Initial Inventory` | `Product Name` starts with `"S700_"` |

The parameters \(r_i\), \(d_i\), \(s_i\) are taken directly from the corresponding columns of the rows that satisfy the filter. No other data are required.