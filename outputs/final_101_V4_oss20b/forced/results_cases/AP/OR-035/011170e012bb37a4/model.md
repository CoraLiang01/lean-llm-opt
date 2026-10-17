**Mathematical Model**

Let  

- \(I\) be the set of bread types, indexed by the values in column **ProductName** of *file_1_view_0*.  
- For each \(i\in I\)  
  - \(p_i\) = expected profit per unit (column **Value** of *file_1_view_0*).  
  - \(w_i\) = storage weight per unit (column **Weight** of *file_1_view_0*).  
- \(C\) = total storage capacity (column **Capacity** of *file_0_view_0*).  
- Decision variable \(x_i\) = number of units of bread type \(i\) to order (integer, non‑negative).

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column | Notes |
|--------|--------------|--------|-------|
| \(I\) | *file_1_view_0* | **ProductName** | Index set |
| \(p_i\) | *file_1_view_0* | **Value** | Profit per unit |
| \(w_i\) | *file_1_view_0* | **Weight** | Storage weight per unit |
| \(C\) | *file_0_view_0* | **Capacity** | Total storage capacity |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.