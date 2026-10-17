## Mathematical Model

**Index set**  
\(I=\{1,\dots ,10\}\)  (VehicleIDs)

**Parameters**  
\(\text{Cap}_i\) – daily inventory limit for vehicle \(i\) (from *capacity.csv*)  
\(\text{Val}_i\) – benefit coefficient for vehicle \(i\) (from *products.csv*)

**Decision variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of vehicles of type \(i\) ordered per day

**Objective**  
\[
\max \;\sum_{i\in I} \text{Val}_i\,x_i
\]

**Constraints**

1. **Per‑type limit**  
\[
x_i \;\le\; \text{Cap}_i \qquad \forall i\in I
\]

2. **Total inventory capacity**  
\[
\sum_{i\in I} x_i \;\le\; \sum_{i\in I} \text{Cap}_i
\]

---

## Data Mapping

| Table ID | Source File | Columns Used | Mapping Rule |
|----------|-------------|--------------|--------------|
| `file_0_view_0` | `capacity.csv` | `VehicleID`, `VehicleType`, `Capacity` | For each \(i\in I\): \(\text{Cap}_i = \text{Capacity}\) where `VehicleID` = \(i\). |
| `file_1_view_0` | `products.csv` | `ProductName`, `Value`, `Weight` | For each \(i\in I\): \(\text{Val}_i = \text{Value}\) where `ProductName` = `VehicleType` of the same `VehicleID` \(i\). |

*No other tables or columns are referenced.*