**Mathematical model**

Let  

- \(I=\{1,\dots ,10\}\) be the set of vehicle types (VehicleID).  
- \(x_i\in \mathbb{Z}_{+}\) be the number of vehicles of type \(i\) ordered per day.  
- \(cap_i\) be the daily inventory limit for type \(i\).  
- \(b_i\) be the benefit coefficient for type \(i\).  

\[
\begin{aligned}
\max\;&\sum_{i\in I} b_i\,x_i \\[4pt]
\text{s.t. } &x_i \;\le\; cap_i , &&\forall i\in I \\[4pt]
             &\sum_{i\in I} x_i \;\le\; \sum_{i\in I} cap_i .
\end{aligned}
\]

All variables are non‑negative integers.

---

**Data mapping**

| Parameter | Source table | Column name | Notes |
|-----------|--------------|-------------|-------|
| \(I\) | `file_0_view_0` | `VehicleID` | Index set |
| \(cap_i\) | `file_0_view_0` | `Capacity` | Daily inventory limit per type |
| \(b_i\) | `file_1_view_0` | `Value` | Benefit coefficient per type |

The total capacity constraint uses the sum of all `cap_i` values from `file_0_view_0`.