**Mathematical Model**

Let  

- \(I\) be the set of vehicle types.  
- \(b_i\) be the benefit coefficient for vehicle type \(i\).  
- \(c_i\) be the daily inventory limit for vehicle type \(i\).  
- \(C_{\text{tot}}=\sum_{i\in I}c_i\) be the total inventory capacity.  
- \(x_i\) be the number of units of vehicle type \(i\) ordered per day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} b_i\,x_i \\[4pt]
\text{s.t. } & x_i \;\le\; c_i, &&\forall i\in I \quad\text{(per‑type capacity)}\\[2pt]
& \sum_{i\in I} x_i \;\le\; C_{\text{tot}}, &&\text{(total capacity)}\\[2pt]
& x_i \;\in\; \mathbb{Z}_{+}, &&\forall i\in I \quad\text{(integer, non‑negative)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Matching Key |
|-----------|--------------|---------------|--------------|
| \(b_i\) | `file_1_view_0` | `Value` | `ProductName = i` |
| \(c_i\) | `file_0_view_0` | `Capacity` | `VehicleType = i` |
| \(C_{\text{tot}}\) | derived | – | \(C_{\text{tot}} = \sum_{i\in I} c_i\) |

All other data (e.g., the set \(I\)) are obtained by the union of the vehicle type identifiers present in the two source tables. No numerical values are listed; the model is fully symbolic.