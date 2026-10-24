## Mathematical Model

**Index set**  
\(I=\{\text{Sedans},\text{SUVs},\text{Electric Vehicles},\text{Hybrid Vehicles},\text{Trucks},\text{Sports Cars},\text{Compact Cars},\text{Luxury Sedans},\text{Vans},\text{Pickup Trucks}\}\)

**Parameters**  
\[
\begin{aligned}
\text{cap}_i &\;=\; \text{Capacity of vehicle type }i \quad (i\in I)\\
\text{val}_i &\;=\; \text{Benefit coefficient of vehicle type }i \quad (i\in I)\\
C_{\text{tot}} &\;=\; \sum_{i\in I}\text{cap}_i
\end{aligned}
\]

**Decision variables**  
\[
x_i \in \mathbb{Z}_{\ge 0}\qquad (i\in I)
\]

**Objective**  
\[
\max \;\sum_{i\in I}\text{val}_i\,x_i
\]

**Constraints**  

1. **Per‑type capacity limits**  
\[
x_i \;\le\; \text{cap}_i \qquad \forall i\in I
\]

2. **Total inventory capacity**  
\[
\sum_{i\in I}x_i \;\le\; C_{\text{tot}}
\]

---

## Data Mapping

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(\text{cap}_i\) | `file_0_view_0` | `VehicleType`, `Capacity` | For each record, set \(\text{cap}_{\text{VehicleType}} = \text{Capacity}\). |
| \(\text{val}_i\) | `file_1_view_0` | `ProductName`, `Value` | For each record, set \(\text{val}_{\text{ProductName}} = \text{Value}\). |
| \(C_{\text{tot}}\) | Derived | – | Compute as \(\sum_{i\in I}\text{cap}_i\). |

All indices \(i\) are taken from the union of `VehicleType` and `ProductName` values; the two lists are identical in the supplied data. All variables \(x_i\) are non‑negative integers. The model is a pure integer linear program.