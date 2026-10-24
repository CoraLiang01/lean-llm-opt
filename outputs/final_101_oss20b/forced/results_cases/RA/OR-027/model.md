**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Sub Category* contains the string “Organ”.  
- For each \(i\in I\)  
  - \(r_i\)  – revenue per unit (from column **Revenue**)  
  - \(d_i\)  – demand quantity (from column **Demand**)  
  - \(s_i\)  – initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to be fulfilled.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le d_i && \forall i\in I \\[2pt]
             & 0 \le x_i \le s_i && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Sub Category` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

All parameter values are taken directly from the rows returned by the query; no additional data are introduced.