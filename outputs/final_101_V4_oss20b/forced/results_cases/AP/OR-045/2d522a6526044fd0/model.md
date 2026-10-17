**Mathematical Model**

Let  

- \(I\) be the set of product types.  
- \(w_i\) be the weight (in the same units as the capacity) of one unit of product \(i\).  
- \(v_i\) be the benefit (value) obtained from one unit of product \(i\).  
- \(C\) be the overall inventory‑capacity.  
- \(x_i\) be the integer number of units of product \(i\) ordered daily.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` |
| \(w_i\) | Weight of product \(i\) | `file_1_view_0` | `Weight` |
| \(v_i\) | Value of product \(i\) | `file_1_view_0` | `Value` |
| \(I\) | Product names | `file_1_view_0` | `ProductName` |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.