**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \; v_i\,x_i \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} x_i \;\le\; C_{\text{tot}} \\[4pt]
& x_i \;\le\; c_i , \qquad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{+}, \qquad \forall i\in I
\end{aligned}
\]

where  

* \(I\) – set of vehicle types (indexed by `VehicleID`).  
* \(c_i\) – daily inventory limit for type \(i\).  
* \(v_i\) – benefit coefficient for type \(i\).  
* \(C_{\text{tot}} = \sum_{i\in I} c_i\).

---

**Data Mapping Rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(I\) | `file_0_view_0` | `VehicleID` |
| \(c_i\) | `file_0_view_0` | `Capacity` (row where `VehicleID` = i) |
| \(v_i\) | `file_1_view_0` | `Value` (row where `ProductName` = `VehicleType` of i) |
| \(C_{\text{tot}}\) | Derived | Sum of all \(c_i\) |

All numeric values are taken directly from the corresponding CSV records; no enumeration of individual rows is shown.