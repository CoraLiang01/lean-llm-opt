**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product_Reference* starts with **ELE‑S**.  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – demand quantity (from column **Demand**)  
  - \(s_i\)  – initial inventory (from column **Initial Inventory**)  
  - \(x_i\)  – number of units of product \(i\) to be sold (decision variable)

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& 0 \le x_i \le d_i && \forall i\in I \\[2pt]
& 0 \le x_i \le s_i && \forall i\in I \\[2pt]
& x_i \text{ continuous} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(s_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index set \(I\) | `file_0_view_0` | `Product_Reference` (filter: prefix `"ELE-S"`) |

All parameters are taken directly from the specified columns of the table `file_0_view_0`. No additional data or aggregation is introduced.