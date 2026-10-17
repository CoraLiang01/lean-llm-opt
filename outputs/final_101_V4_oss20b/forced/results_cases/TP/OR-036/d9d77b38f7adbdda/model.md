**Mathematical Model**

Let  

- \(I\) be the set of vehicle types (product names).  
- \(v_i\) be the benefit coefficient of product \(i\).  
- \(w_i\) be the weight (inventory consumption) of product \(i\).  
- \(C\) be the total inventory capacity.  
- \(x_i\) be the number of units of product \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(C\) | `file_0_view_0` | `Capacity` | Single‑row table |
| \(v_i\) | `file_1_view_0` | `Value` | Row where `ProductName = i` |
| \(w_i\) | `file_1_view_0` | `Weight` | Row where `ProductName = i` |
| \(I\) | `file_1_view_0` | `ProductName` | All distinct product names |

These mappings bind each model parameter to the exact source data provided in the observation. No additional data or constraints are introduced.