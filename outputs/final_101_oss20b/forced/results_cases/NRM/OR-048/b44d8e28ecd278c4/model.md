**Abstract Model**

**Index Sets**  
\(I\) – set of storage areas (indexed by StorageID)  
\(J\) – set of air‑conditioner product types (indexed by ProductName)

**Parameters**  
\(\text{Capacity}_{i}\) – capacity of storage area \(i\) (units of size)  
\(\text{Value}_{j}\) – monetary value of one unit of product \(j\)  
\(\text{Weight}_{j}\) – size (weight) of one unit of product \(j\)

**Decision Variables**  
\(x_{ij}\) – number of units of product \(j\) placed in storage area \(i\)  
\(x_{ij}\in \mathbb{Z}_{\ge 0}\)

**Objective Function**  
\[
\max \sum_{i\in I}\sum_{j\in J} \text{Value}_{j}\;x_{ij}
\]

**Constraints**

1. **Capacity Constraints (per storage area)**  
\[
\sum_{j\in J} \text{Weight}_{j}\;x_{ij}\;\le\;\text{Capacity}_{i},
\qquad \forall i\in I
\]

2. **Non‑negativity / Integer Constraints**  
\[
x_{ij}\in \mathbb{Z}_{\ge 0},\qquad \forall i\in I,\; j\in J
\]

---

### Data Mapping

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(\text{Capacity}_{i}\) | `capacity.csv` | `Capacity` |
| \(\text{Capacity}_{i}\) | `capacity.csv` | `StorageID` |
| \(\text{Value}_{j}\) | `products.csv` | `Value` |
| \(\text{Value}_{j}\) | `products.csv` | `ProductName` |
| \(\text{Weight}_{j}\) | `products.csv` | `Weight` |
| \(\text{Weight}_{j}\) | `products.csv` | `ProductName` |

*All table identifiers are the exact `table_id` values returned by the CSVQA query (`file_0_view_0` for `capacity.csv` and `file_1_view_0` for `products.csv`). No record values or counts are reproduced.*