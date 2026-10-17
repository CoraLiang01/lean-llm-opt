**Mathematical Model**

Let  

- \(V\) be the set of vehicle types.  
- \(x_v\) – number of vehicles of type \(v\) ordered per day.  

Parameters  

- \(\text{cap}_v\) – daily inventory limit for type \(v\).  
- \(\text{benefit}_v\) – benefit coefficient for type \(v\).  

\[
\begin{aligned}
\max\;& \sum_{v\in V} \text{benefit}_v\,x_v \\[4pt]
\text{s.t.}\;& x_v \;\le\; \text{cap}_v , &&\forall v\in V \\[2pt]
& \sum_{v\in V} x_v \;\le\; \sum_{v\in V} \text{cap}_v \\[2pt]
& x_v \in \mathbb{Z}_{\ge 0}, &&\forall v\in V
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(\text{cap}_v\) | `file_0_view_0` | `VehicleType` | `Capacity` |
| \(\text{benefit}_v\) | `file_1_view_0` | `ProductName` | `Value` |

The index set \(V\) is the union of all distinct `VehicleType` values in `file_0_view_0` (which also match the `ProductName` values in `file_1_view_0`).