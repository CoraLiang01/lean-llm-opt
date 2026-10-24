**Mathematical Model**

Let  

- \(I\) be the set of Aalop products (rows returned by the filter).  
- For each \(i\in I\)  
  - \(r_i\) : revenue per unit (from column **Revenue**)  
  - \(d_i\) : demand (from column **Demand**)  
  - \(v_i\) : initial inventory (from column **Initial Inventory**)  
- Decision variable  
  - \(x_i\) : units of product \(i\) to sell (integer, but can be relaxed to continuous if desired).

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le \min\{d_i,\;v_i\} && \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

(If integer units are not required, replace \(\mathbb{Z}_{+}\) by \(\mathbb{R}_{+}\).)

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set of returned rows | `file_0_view_0` | `Product Name` (filtered by prefix “Aalop”) |
| \(r_i\) | Revenue per unit | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(v_i\) | Initial inventory | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No additional numerical values are introduced.