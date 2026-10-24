**Mathematical Model**

Let  

- \(I\) be the set of product references whose `Product_Reference` starts with **ELE‑S**.  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – demand quantity (from column **Demand**)  
  - \(v_i\)  – initial inventory (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i\) – number of units of product \(i\) to sell (integer, \(\ge 0\))  

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le v_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column | Parameter | Mapping Rule |
|----------|--------|-----------|--------------|
| `file_0_view_0` | `Product_Reference` | index \(i\) | each distinct value forms an element of \(I\) |
| `file_0_view_0` | `Revenue` | \(r_i\) | \(r_i =\) value in **Revenue** column for product \(i\) |
| `file_0_view_0` | `Demand` | \(d_i\) | \(d_i =\) value in **Demand** column for product \(i\) |
| `file_0_view_0` | `Initial Inventory` | \(v_i\) | \(v_i =\) value in **Initial Inventory** column for product \(i\) |

All parameters are taken directly from the rows returned by the filter that selects `Product_Reference` values beginning with `"ELE-S"`. No additional numerical values or constraints are introduced beyond those specified.