**Mathematical Model**

Let  

- \(I\) – set of vehicle types (indexed by the values in *ProductName*).  
- \(C\) – total inventory capacity (from *Capacity*).  
- \(v_i\) – benefit coefficient of vehicle type \(i\) (from *Value*).  
- \(w_i\) – weight (inventory consumption) of vehicle type \(i\) (from *Weight*).  
- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of vehicle type \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` |
| \(I\) | Product names | `file_1_view_0` | `ProductName` |
| \(v_i\) | Benefit coefficient | `file_1_view_0` | `Value` |
| \(w_i\) | Weight (inventory usage) | `file_1_view_0` | `Weight` |

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.