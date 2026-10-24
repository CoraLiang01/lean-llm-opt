**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{Value}_i \; x_i \\[4pt]
\text{subject to}\quad
& x_i \;\le\; \text{Capacity}_i && \forall i\in I \\[2pt]
& \sum_{i\in I} x_i \;\le\; \text{TotalCapacity} \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} && \forall i\in I
\end{aligned}
\]

**Index Set**

- \(I\): set of vehicle types (identified by the values in the columns *VehicleType* and *ProductName*).

**Parameters**

- \(\text{Value}_i\): benefit coefficient for vehicle type \(i\).
- \(\text{Capacity}_i\): daily inventory limit for vehicle type \(i\).
- \(\text{TotalCapacity}\): total inventory capacity, defined as \(\displaystyle \sum_{i\in I}\text{Capacity}_i\).

**Variables**

- \(x_i\): number of vehicles of type \(i\) ordered per day (integer, non‑negative).

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(\text{Value}_i\) | `file_1_view_0` | `ProductName` | `Value` |
| \(\text{Capacity}_i\) | `file_0_view_0` | `VehicleType` | `Capacity` |
| \(\text{TotalCapacity}\) | *derived* | – | \(\sum_{i}\text{Capacity}_i\) |

All other data are implicit in the index set \(I\) and the parameter definitions above.