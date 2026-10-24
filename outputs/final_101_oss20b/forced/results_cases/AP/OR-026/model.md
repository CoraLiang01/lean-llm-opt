**Mathematical Model**

Let  

- \(I\) be the set of Fashion products extracted from the filtered table.  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(inv_i\) : initial inventory (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i\) : number of units of product \(i\) to sell (integer, \(\ge 0\))

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le \min\{d_i,\;inv_i\}\quad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Set of product identifiers | `file_0_view_0` | `Product Name` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(inv_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – | – |

All parameters are taken directly from the filtered rows of the CSV file; no numerical values are enumerated in the model.