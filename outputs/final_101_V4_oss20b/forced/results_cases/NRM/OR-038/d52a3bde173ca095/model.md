**Mathematical Model**

Let  

* \(I\) – index set of vehicle types.  
* \(b_i\) – benefit coefficient of type \(i\) (from *products.csv*).  
* \(c_i\) – daily inventory limit of type \(i\) (from *capacity.csv*).  
* \(C_{\text{tot}}\) – total inventory capacity, \(C_{\text{tot}}=\sum_{i\in I}c_i\).  
* \(x_i\) – number of vehicles of type \(i\) ordered per day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} b_i\,x_i \\[4pt]
\text{s.t. } & 0 \le x_i \le c_i , && \forall i\in I \\[4pt]
& \sum_{i\in I} x_i \le C_{\text{tot}} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}, && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(c_i\)   | `file_0_view_0` | `Capacity` |
| \(b_i\)   | `file_1_view_0` | `Value` |
| \(C_{\text{tot}}\) | derived | sum of `Capacity` over all rows in `file_0_view_0` |

*Index set \(I\) is the set of all `VehicleType` values appearing in `file_0_view_0` (and matched by `ProductName` in `file_1_view_0`).*