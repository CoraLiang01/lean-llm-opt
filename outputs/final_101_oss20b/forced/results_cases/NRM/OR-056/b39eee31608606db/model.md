**Abstract Model**

**Sets**  
\(I\) – set of display areas (indexed by DisplayID).  
\(J\) – set of vessel types (indexed by ProductName).

**Parameters**  
\(\text{Capacity}_i\) – capacity of display area \(i\).  
\(\text{Value}_j\) – monetary value of vessel type \(j\).  
\(\text{Weight}_j\) – size (weight) of vessel type \(j\).

**Decision Variables**  
\(x_{ij}\) – number of vessels of type \(j\) placed in display area \(i\).  
\(x_{ij}\in \mathbb{Z}_{\ge 0}\).

**Objective Function**  
\[
\max \sum_{i\in I}\sum_{j\in J} \text{Value}_j \; x_{ij}
\]

**Constraints**  

1. **Capacity Constraints** – for each display area \(i\):
\[
\sum_{j\in J} \text{Weight}_j \; x_{ij} \;\le\; \text{Capacity}_i
\]

2. **Non‑negativity / Integer** – for all \(i\in I,\; j\in J\):
\[
x_{ij}\in \mathbb{Z}_{\ge 0}
\]

---

### Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

*No record values or counts are reproduced; only the table identifiers and column names are shown.*