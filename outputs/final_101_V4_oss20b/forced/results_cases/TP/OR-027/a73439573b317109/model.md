**Mathematical Model**

Let  

- \(I=\{\text{Organic Fruits},\text{Organic Staples},\text{Organic Vegetables}\}\)  
- \(x_i\) – units of product \(i\) to be sold, \(x_i\ge 0\)  

Parameters (obtained from the filtered view):

- \(r_i\) – revenue per unit of product \(i\)  
- \(d_i\) – demand for product \(i\)  
- \(s_i\) – initial inventory of product \(i\)

\[
\begin{aligned}
\max\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;&0 \le x_i \le d_i &&\forall i\in I \\[2pt]
&0 \le x_i \le s_i &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Table ID | Column | Parameter | Index Mapping |
|----------|--------|-----------|---------------|
| `file_0_view_0` | `Revenue` | \(r_i\) | `Sub Category` → \(i\) |
| `file_0_view_0` | `Demand` | \(d_i\) | `Sub Category` → \(i\) |
| `file_0_view_0` | `Initial Inventory` | \(s_i\) | `Sub Category` → \(i\) |

The index set \(I\) is derived from the distinct values of the `Sub Category` column in the filtered view (`Organic Fruits`, `Organic Staples`, `Organic Vegetables`). All parameter values are taken directly from the corresponding columns of the same rows.