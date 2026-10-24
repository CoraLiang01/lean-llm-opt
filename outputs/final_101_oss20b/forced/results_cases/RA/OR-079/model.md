**Mathematical Model**

Let  

* \(F=\{A1,\dots ,A15\}\) – set of potential factory sites  
* \(D=\{B1,\dots ,B8\}\) – set of distribution centers  

Parameters  

* \(\text{FixedCost}_f\) – fixed construction cost of facility \(f\)  
* \(\text{Capacity}_f\) – maximum production units that facility \(f\) can supply  
* \(\text{ShipCost}_{f,d}\) – variable shipping cost per unit from \(f\) to \(d\)  
* \(\text{Demand}_d\) – required units at destination \(d\)

Decision variables  

* \(y_f \in \{0,1\}\) – 1 if facility \(f\) is built, 0 otherwise  
* \(x_{f,d}\in \mathbb{Z}_{\ge 0}\) – number of units shipped from \(f\) to \(d\)

---

\[
\begin{aligned}
\min\;& \sum_{f\in F}\text{FixedCost}_f\,y_f
      \;+\;\sum_{f\in F}\sum_{d\in D}\text{ShipCost}_{f,d}\,x_{f,d} \\[4pt]
\text{s.t. } & \sum_{d\in D} x_{f,d}\;\le\; \text{Capacity}_f\,y_f,
      &&\forall f\in F \quad\text{(facility capacity)}\\[4pt]
& \sum_{f\in F} x_{f,d}\;\ge\; \text{Demand}_d,
      &&\forall d\in D \quad\text{(demand satisfaction)}\\[4pt]
& x_{f,d}\;\ge\;0,\;x_{f,d}\in\mathbb{Z},
      &&\forall f\in F,\;d\in D\\[2pt]
& y_f\in\{0,1\},
      &&\forall f\in F
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(\text{FixedCost}_f\) | `file_0_view_0` | `Facility` | `FixedCost` |
| \(\text{Capacity}_f\) | `file_0_view_0` | `Facility` | `Capacity` |
| \(\text{ShipCost}_{f,d}\) | `file_1_view_0` | `Origin` | `B1`–`B8` (column name equals destination ID) |
| \(\text{Demand}_d\) | `file_2_view_0` | `Destination` | `Demand` |

All indices \(f\) and \(d\) are taken directly from the corresponding ID columns in the source tables. No additional data are introduced beyond what is supplied.