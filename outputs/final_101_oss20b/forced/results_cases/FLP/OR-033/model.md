**Mathematical Model**

Let  

* \(I\) – set of all “Baby” products that appear in the filtered view.  
* For each \(i\in I\)  
  * \(r_i\) – revenue per unit (from column **Revenue**).  
  * \(d_i\) – demand quantity (from column **Demand**).  
  * \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

* \(x_i \ge 0\) – number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le s_i && \forall i\in I
\end{aligned}
\]

(If desired, the two upper‑bound constraints can be combined as  
\(x_i \le \min\{d_i,s_i\}\).)

---

**Data Mapping**

| Table ID | Column in CSV | Parameter / Index |
|----------|---------------|-------------------|
| `file_0_view_0` | `Product Name` | maps to index \(i\) |
| `file_0_view_0` | `Revenue` | \(r_i\) |
| `file_0_view_0` | `Demand` | \(d_i\) |
| `file_0_view_0` | `Initial Inventory` | \(s_i\) |

The mapping uses the filtered view (`file_0_view_0`) that contains only rows where **Product Name** contains the substring `"Baby"`. No other tables or columns are referenced.