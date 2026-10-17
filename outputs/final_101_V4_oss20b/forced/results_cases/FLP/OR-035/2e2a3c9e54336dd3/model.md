**Mathematical Model**

Let  

- \(I\) be the set of bread types.  
- \(x_i \in \mathbb{Z}_{+}\) be the number of units of type \(i\) ordered.  
- \(v_i\) be the expected profit per unit of type \(i\).  
- \(w_i\) be the storage weight per unit of type \(i\).  
- \(C\) be the total storage capacity.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Mapping |
|-----------|--------------|--------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | scalar capacity |
| \(v_i\) | `file_1_view_0` | `Value` | profit per unit |
| \(w_i\) | `file_1_view_0` | `Weight` | weight per unit |
| Product index \(i\) | `file_1_view_0` | `ProductName` | identifier for each bread type |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.