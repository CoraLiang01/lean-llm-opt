##### Abstract Model

**Index Sets**  
\(I\) – set of sections (indexed by SectionID)  
\(J\) – set of products (indexed by ProductName)

**Parameters**  
\(C_i\) – capacity of section \(i\) (units of shelf space)  
\(V_j\) – revenue (value) per unit of product \(j\)  
\(W_j\) – shelf‑space requirement per unit of product \(j\)

**Decision Variables**  
\(x_{ij}\) – number of units of product \(j\) stocked in section \(i\)  

\(x_{ij}\in \mathbb{Z}_{\ge 0}\)

**Objective Function**  

\[
\max \sum_{i\in I}\sum_{j\in J} V_j \, x_{ij}
\]

**Constraints**

1. **Capacity Constraints**  

\[
\sum_{j\in J} W_j \, x_{ij} \;\le\; C_i , \qquad \forall i\in I
\]

2. **Non‑negativity / Integer Constraints**  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}, \qquad \forall i\in I,\; j\in J
\]

---

##### Data Mapping

| Parameter | Source Table ID | Column Name |
|-----------|-----------------|-------------|
| \(C_i\)   | file_0_view_0   | Capacity    |
| \(V_j\)   | file_1_view_0   | Value       |
| \(W_j\)   | file_1_view_0   | Weight      |

*No record values or counts are reproduced; only the exact table identifiers and column names are listed.*