**Mathematical Model**

Let  

* \(I\) be the set of all products whose name starts with “Books”.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**)  
  * \(d_i\) – demand (from column **Demand**)  
  * \(s_i\) – initial inventory (from column **Initial Inventory**)  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; d_i &&\forall i\in I \\[2pt]
             & x_i \;\le\; s_i &&\forall i\in I \\[2pt]
             & x_i \;\ge\; 0   &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\)   | revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\)   | demand quantity | `file_0_view_0` | `Demand` |
| \(s_i\)   | initial inventory | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | product identifier | `file_0_view_0` | `Product_Name` (rows 0‑4) |

*Each row of `file_0_view_0` after applying the filter “Product_Name starts with ‘Books’” defines a distinct index \(i\). The mapping preserves the original row order (source_row 0 to 4). No additional data or aggregation is introduced.*