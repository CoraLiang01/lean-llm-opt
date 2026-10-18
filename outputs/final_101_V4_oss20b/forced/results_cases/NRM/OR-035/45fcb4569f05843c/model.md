##### Abstract Model

**Index Set**  
\(I\) – set of bread types.

**Parameters**  
\(p_i\) – expected profit per unit of bread type \(i\).  
\(w_i\) – weight (space consumption) per unit of bread type \(i\).  
\(C\) – total storage capacity available.

**Decision Variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of bread type \(i\) to order.

**Objective Function**  
\[
\max \sum_{i \in I} p_i \, x_i
\]

**Constraints**  

1. **Capacity Constraint**  
\[
\sum_{i \in I} w_i \, x_i \;\le\; C
\]

2. **Non‑negativity and Integer Requirement**  
\[
x_i \in \mathbb{Z}_{\ge 0}, \qquad \forall i \in I
\]

---

##### Data Mapping

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(p_i\)   | `file_1_view_0` | `Value`       |
| \(w_i\)   | `file_1_view_0` | `Weight`      |
| \(C\)     | `file_0_view_0` | `Capacity`    |

*No record values or counts are reproduced; only the table identifiers and column names are shown.*