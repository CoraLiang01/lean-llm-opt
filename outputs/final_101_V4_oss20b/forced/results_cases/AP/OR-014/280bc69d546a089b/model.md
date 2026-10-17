**Mathematical Model**

Let  

- \(I\) be the set of pizza types (rows of the table).  
- For each \(i\in I\)  
  - \(r_i\) – revenue per unit (from column **Revenue**).  
  - \(d_i\) – demand (from column **Demand**).  
  - \(v_i\) – initial inventory (from column **Initial Inventory**).  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – units of pizza type \(i\) to fulfill.

\[
\begin{aligned}
\max\;& \sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. } & x_i \le d_i && \forall i\in I \\[2pt]
             & x_i \le v_i && \forall i\in I \\[2pt]
             & x_i \in \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(r_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(v_i\)   | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is the set of all `Product Name` values in `file_0_view_0`.