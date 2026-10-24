**Mathematical Model**

Let  

- \(I\) : set of dairy products (indexed by the product name).  
- \(r_i\) : revenue per unit of product \(i\).  
- \(d_i\) : deterministic demand for product \(i\).  
- \(inv_i\) : initial inventory available for product \(i\).  
- \(x_i\) : number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le inv_i && \forall i\in I
\end{aligned}
\]

All variables \(x_i\) are continuous (non‑negative).

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | product identifiers | `file_0_view_0` | `Full_Product_Name` |
| \(r_i\) | revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | demand | `file_0_view_0` | `Demand` |
| \(inv_i\) | initial inventory | `file_0_view_0` | `Initial Inventory` |

No additional parameters or constraints are required.