##### Abstract Model

**Index Set**  
\(I\) – set of vehicle types.

**Parameters**  
\(p_i\) – profit per unit of vehicle type \(i\).  
\(w_i\) – weight (inventory consumption) per unit of vehicle type \(i\).  
\(C\) – total inventory capacity.

**Decision Variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of vehicles of type \(i\) to order per day.

**Objective Function**  
\[
\max \sum_{i \in I} p_i \, x_i
\]

**Constraints**  

1. **Capacity Constraint**  
\[
\sum_{i \in I} w_i \, x_i \;\le\; C
\]

2. **Non‑negativity and Integer Domain**  
\[
x_i \in \mathbb{Z}_{\ge 0}, \quad \forall i \in I
\]

---

##### Data Mapping

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(p_i\)   | file_1_view_0   | Value         |
| \(w_i\)   | file_1_view_0   | Weight        |
| \(C\)     | file_0_view_0   | Capacity      |

*No record values or counts are reproduced.*