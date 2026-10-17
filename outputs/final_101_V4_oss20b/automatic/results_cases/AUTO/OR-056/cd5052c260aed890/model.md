**Model**

Let  

- \(I=\{\,i\mid \text{DisplayID}=i\text{ in } \text{file\_0\_view\_0}\,\}\)  
- \(J=\{\,j\mid \text{ProductName}=j\text{ in } \text{file\_1\_view\_0}\,\}\)

Parameters  

- \(C_i\) – capacity of display \(i\)  (\(C_i\) from column **Capacity** of *file_0_view_0*)  
- \(V_j\) – value of product \(j\)  (\(V_j\) from column **Value** of *file_1_view_0*)  
- \(W_j\) – weight (size) of product \(j\)  (\(W_j\) from column **Weight** of *file_1_view_0*)

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed in display \(i\)

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} V_j\,x_{ij}
\]

---

**Constraints**

1. **Capacity of each display**

\[
\sum_{j\in J} W_j\,x_{ij}\;\le\; C_i ,\qquad \forall\, i\in I
\]

2. **Non‑negativity and integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0},\qquad \forall\, i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(C_i\)   | file_0_view_0 | Capacity |
| \(V_j\)   | file_1_view_0 | Value |
| \(W_j\)   | file_1_view_0 | Weight |

All index sets \(I\) and \(J\) are taken directly from the rows of the respective tables. No additional constraints or data are introduced beyond those specified.