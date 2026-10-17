**Mathematical Model**

Let  

- \(I=\{\,\text{ProductName}\,\}\) be the set of all products listed in *products.csv*.  
- \(x_i\ge 0\) : number of units of product \(i\) ordered each day.  

Parameters (all obtained directly from the CSV files)

| Parameter | Symbol | Source table | Source column |
|-----------|--------|--------------|---------------|
| Product weight | \(w_i\) | file_1_view_0 | Weight |
| Product value | \(v_i\) | file_1_view_0 | Value |
| Total storage capacity | \(C\) | file_0_view_0 | Capacity |

The optimization problem is

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C, \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I .
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Table ID | Column in model | CSV column | Notes |
|----------|-----------------|------------|-------|
| file_0_view_0 | \(C\) | Capacity | Single‑row table |
| file_1_view_0 | \(w_i\) | Weight | Indexed by ProductName |
| file_1_view_0 | \(v_i\) | Value | Indexed by ProductName |

All parameters are taken verbatim from the corresponding CSV columns; no numerical values are enumerated in the model.