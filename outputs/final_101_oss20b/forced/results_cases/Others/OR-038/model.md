## Mathematical Model

**Indices**  
\(I=\{1,\dots ,10\}\) (vehicles, indexed by *VehicleID*)

**Parameters**  
\(\text{cap}_i\) – daily inventory limit for vehicle \(i\) (from *capacity.csv*, column *Capacity*)  
\(\text{val}_i\) – benefit coefficient for vehicle \(i\) (from *products.csv*, column *Value*, matched by *VehicleType* = *ProductName*)

**Decision Variables**  
\(x_i \in \mathbb{Z}_{+}\) – number of vehicles of type \(i\) ordered per day

**Objective**  
\[
\max \;\sum_{i\in I} \text{val}_i\,x_i
\]

**Constraints**  

1. **Per‑type capacity**  
\[
x_i \;\le\; \text{cap}_i \qquad \forall i\in I
\]

2. **Total inventory capacity**  
\[
\sum_{i\in I} x_i \;\le\; \sum_{i\in I} \text{cap}_i
\]

3. **Non‑negativity & integrality**  
\[
x_i \in \mathbb{Z}_{+}\qquad \forall i\in I
\]

---

## Data Mapping

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(\text{cap}_i\) | `file_0_view_0` | `Capacity` | For each row, assign to index \(i=\) *VehicleID* |
| \(\text{val}_i\) | `file_1_view_0` | `Value` | Match rows where *VehicleType* (from `file_0_view_0`) equals *ProductName* (from `file_1_view_0`); assign to same index \(i\) |

All indices \(i\) are taken from the complete set of rows in `file_0_view_0` (10 vehicle types). No additional filtering or aggregation is applied.