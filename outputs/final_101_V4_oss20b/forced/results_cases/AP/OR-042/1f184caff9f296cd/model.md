**Mathematical Model**

Let  

- \(I\) be the set of drug types (product names).  
- \(v_i\) be the benefit coefficient of product \(i\).  
- \(w_i\) be the weight of one unit of product \(i\).  
- \(C\) be the overall inventory capacity.  
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

| Parameter | Source | Table ID | Column(s) | Mapping |
|-----------|--------|----------|-----------|---------|
| \(C\) | Overall capacity | `file_0_view_0` | `Capacity` | \(C = \text{Capacity}\) |
| \(v_i\) | Benefit coefficient | `file_1_view_0` | `Value` | \(v_i = \text{Value}\) for product \(i\) |
| \(w_i\) | Weight per unit | `file_1_view_0` | `Weight` | \(w_i = \text{Weight}\) for product \(i\) |
| \(I\) | Product names | `file_1_view_0` | `ProductName` | \(I = \{\text{ProductName}\}\) |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.