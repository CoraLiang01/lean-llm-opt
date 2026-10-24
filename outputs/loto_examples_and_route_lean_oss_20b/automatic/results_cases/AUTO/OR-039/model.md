## Mathematical Model

**Sets**  
\(I\) – set of vehicle types (Sedans, SUVs, …, Pickup Trucks)  
\(W\) – set of warehouses (Warehouse 1,…,Warehouse 10)

**Parameters**  
\(v_i\) – benefit (value) of one unit of vehicle type \(i\)  
\(w_i\) – weight of one unit of vehicle type \(i\)  
\(C_w\) – capacity of warehouse \(w\)

**Decision Variables**  
\(x_{i,w}\in \mathbb{Z}_{+}\) – number of units of vehicle type \(i\) stored in warehouse \(w\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{w\in W} v_i\,x_{i,w}
\]

**Constraints**  
For every warehouse \(w\in W\):
\[
\sum_{i\in I} w_i\,x_{i,w}\;\le\; C_w
\]

All variables are non‑negative integers.

---

## Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | `file_1_view_0` | `ProductName` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(W\) | `file_0_view_0` | `Warehouse ID` |
| \(C_w\) | `file_0_view_0` | `Capacity` |

All indices \(i\) and \(w\) are taken from the full set of rows returned in the observation; no additional filtering is applied. The model uses only the parameters listed above and respects the integer domain for all decision variables.