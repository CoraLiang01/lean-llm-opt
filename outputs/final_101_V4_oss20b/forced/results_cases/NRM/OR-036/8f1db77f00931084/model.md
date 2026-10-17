**Mathematical Model**

Let  

- \(I\) be the set of vehicle types.  
- For each \(i\in I\):  
  - \(v_i\) – benefit coefficient (value) of vehicle \(i\).  
  - \(w_i\) – inventory weight of vehicle \(i\).  
- \(C\) – total inventory capacity.  
- \(x_i\) – number of units of vehicle \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(I\) | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding columns of the specified tables; no additional data or transformations are introduced.