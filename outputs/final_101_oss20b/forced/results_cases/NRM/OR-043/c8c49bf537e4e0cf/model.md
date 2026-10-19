##### Abstract Model

**Index Set**  
\(I\) – set of drug products.

**Parameters**  
\(v_i\) – benefit value of product \(i\).  
\(w_i\) – weight (or space) requirement of product \(i\).  
\(C\) – total stock capacity of the pharmacy chain.

**Decision Variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to order each day.

**Objective Function**  
\[
\max \sum_{i \in I} v_i \, x_i
\]

**Constraints**  

1. **Capacity Constraint**  
\[
\sum_{i \in I} w_i \, x_i \;\le\; C
\]

2. **Non‑negativity and Integer Requirement**  
\[
x_i \in \mathbb{Z}_{\ge 0}, \quad \forall i \in I
\]

---

##### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(I\) | `file_1_view_0` | `ProductName` (each distinct product) |

*All values are taken directly from the corresponding CSV files; no additional data is introduced.*