**Mathematical Model**

```markdown
**Index set**  
\(I=\{1,2,\dots ,10\}\)  (VehicleID)

**Parameters**  
\(\text{cap}_i\) – daily inventory limit for vehicle type \(i\)  
\(\text{val}_i\) – benefit coefficient for vehicle type \(i\)  
\(C=\sum_{i\in I}\text{cap}_i\) – total inventory capacity

**Decision variables**  
\(x_i\in\mathbb{Z}_{+}\) – number of vehicles of type \(i\) ordered per day

**Objective**  
\[
\max \;\sum_{i\in I}\text{val}_i\,x_i
\]

**Constraints**  

1. **Total capacity**  
\[
\sum_{i\in I}x_i \;\le\; C
\]

2. **Per‑type limits**  
\[
0 \;\le\; x_i \;\le\; \text{cap}_i ,\qquad \forall i\in I
\]
```

---

**Data Mapping Rules**

| Source | Table ID | Column | Mapping to Model |
|--------|----------|--------|------------------|
| `capacity.csv` | `file_0_view_0` | `VehicleID` | Index \(i\) |
| | | `Capacity` | \(\text{cap}_i\) |
| `products.csv` | `file_1_view_0` | `ProductName` | Matches `VehicleType` of same `VehicleID` → index \(i\) |
| | | `Value` | \(\text{val}_i\) |
| | | `Weight` | *unused in the model* |

*The order of records in each table is preserved when assigning indices and parameter values.*