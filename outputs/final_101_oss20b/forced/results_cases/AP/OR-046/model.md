**Mathematical Model**

Let  

- \(I\) be the set of products indexed by the product name.  
- \(v_i\) – benefit (value) per unit of product \(i\).  
- \(w_i\) – weight (space) per unit of product \(i\).  
- \(C\) – total available stock capacity.  
- \(x_i\) – number of units of product \(i\) ordered each day.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Mapping |
|-----------|--------------|---------------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(w_i\) | `file_1_view_0` | `Weight` | `w_i = Weight` |
| \(v_i\) | `file_1_view_0` | `Value` | `v_i = Value` |
| Index set \(I\) | `file_1_view_0` | `ProductName` | `I = {ProductName}` |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.