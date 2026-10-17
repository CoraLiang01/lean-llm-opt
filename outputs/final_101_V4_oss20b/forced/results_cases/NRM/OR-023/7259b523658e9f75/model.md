**Mathematical Model**

Let  

- \(I\) – set of all products whose *Product_Reference* starts with “ELE‑S”.  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**)  
  - \(d_i\) – deterministic demand (from column **Demand**)  
  - \(I_i\) – initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to sell.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t.}\;& x_i \le d_i && \forall i\in I \\[2pt]
& x_i \le I_i && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(I_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | `file_0_view_0` | `Product_Reference` (filtered by prefix “ELE‑S”) |

No additional data or constraints are introduced beyond those specified in the query.