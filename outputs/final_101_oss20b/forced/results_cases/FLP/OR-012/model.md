**Mathematical Model**

Let  

* \(I\) be the set of all products (identified by the *Product Name* column).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from *Revenue*).  
  * \(d_i\) – known demand (from *Demand*).  
  * \(s_i\) – initial inventory (from *Initial Inventory*).  
  * \(x_i\) – number of units of product \(i\) to sell (decision variable).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i &&\forall i\in I \\[2pt]
             & 0 \le x_i \le s_i &&\forall i\in I
\end{aligned}
\]

\(x_i\) are continuous non‑negative variables.

---

**Data Mapping Rules**

| Table ID | Column | Parameter / Index |
|----------|--------|-------------------|
| `file_0_view_0` | `Product Name` | index \(i\) |
| `file_0_view_0` | `Revenue` | \(r_i\) |
| `file_0_view_0` | `Demand` | \(d_i\) |
| `file_0_view_0` | `Initial Inventory` | \(s_i\) |

All 119 rows of `file_0_view_0` are used to populate the sets and parameters. No additional constraints or capacities are introduced beyond the demand and inventory limits.