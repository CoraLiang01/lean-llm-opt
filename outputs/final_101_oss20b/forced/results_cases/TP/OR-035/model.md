**Mathematical Model**

Let  

- \(I\) be the set of bread types (indexed by the product names).  
- \(x_i\) be the number of units of bread type \(i\) ordered each day.  

Parameters  

- \(v_i\) – expected profit per unit of bread type \(i\).  
- \(w_i\) – weight (storage requirement) per unit of bread type \(i\).  
- \(C\) – total storage capacity available each day.  

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(i\) | `file_1_view_0` | `ProductName` | `i = ProductName` |
| \(v_i\) | `file_1_view_0` | `Value` | `v_i = Value` |
| \(w_i\) | `file_1_view_0` | `Weight` | `w_i = Weight` |

All indices, parameters, and variables are defined exactly as above; no additional data or constraints are introduced.