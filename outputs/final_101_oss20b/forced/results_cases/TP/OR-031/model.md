**Mathematical Model**

Let  

* \(I\) be the set of dairy products (identified by *Full_Product_Name*).  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – maximum demand (from column **Demand**).  
  * \(s_i\) – initial inventory limit (from column **Initial Inventory**).  
* Decision variable  
  * \(x_i\ge 0\) – number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le s_i && \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Table ID | Column | Parameter / Index |
|----------|--------|-------------------|
| `file_0_view_0` | `Full_Product_Name` | index set \(I\) (product identifier) |
| `file_0_view_0` | `Revenue` | \(r_i\) |
| `file_0_view_0` | `Demand` | \(d_i\) |
| `file_0_view_0` | `Initial Inventory` | \(s_i\) |

All numeric values are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or aggregation is required.