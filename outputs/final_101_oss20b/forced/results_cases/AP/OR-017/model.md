**Symbolic Model**

Let  

- \(I\) be the set of product identifiers whose SKU starts with “ZZ”.  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand quantity (from column **Demand**)  
  - \(inv_i\) : initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i\) : number of units of product \(i\) to sell (units to fulfill)

\[
\begin{aligned}
\max \quad & \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\quad & 0 \le x_i \le \min\{d_i,\;inv_i\}\qquad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{+}\qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Set of SKUs with prefix “ZZ” | `file_0_view_0` | `SKU` |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand quantity | `file_0_view_0` | `Demand` |
| \(inv_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – | – |

All parameters are taken directly from the filtered rows of the CSV file; no numerical values are enumerated in the model.