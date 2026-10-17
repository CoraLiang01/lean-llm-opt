**Mathematical Model**

Let  

- \(I\) – set of all products in the table.  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**).  
  - \(d_i\) – total demand (from column **Demand**).  
  - \(s_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i \in \mathbb{Z}_{+}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \le d_i &&\forall i\in I \\[2pt]
& x_i \le s_i &&\forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{+} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) consists of all rows returned in `file_0_view_0` (109 products). No additional constraints or capacities are imposed beyond the per‑product demand and inventory limits.