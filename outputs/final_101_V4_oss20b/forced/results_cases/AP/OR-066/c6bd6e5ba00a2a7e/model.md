**Mathematical Model**

Let  

- \(I=\{S1,S2\}\) : set of suppliers  
- \(J=\{C1,C2\}\) : set of supermarkets  

Parameters  

- \(f_i\) : fixed activation cost of supplier \(i\)  \(i\in I\)  
- \(d_j\) : demand of supermarket \(j\)  \(j\in J\)  
- \(c_{ij}\) : per‑unit transportation cost from supplier \(i\) to supermarket \(j\) \(i\in I,\;j\in J\)  

Decision variables  

- \(y_i\in\{0,1\}\) : 1 if supplier \(i\) is activated, 0 otherwise  
- \(x_{ij}\ge 0\) : quantity supplied from supplier \(i\) to supermarket \(j\)  

---

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

---

**Constraints**

1. **Demand satisfaction**

\[
\sum_{i\in I} x_{ij}=d_j \quad \forall j\in J
\]

2. **Supplier activation linkage**

\[
x_{ij}\;\le\; M\,y_i \quad \forall i\in I,\; j\in J
\]

where \(M=\sum_{j\in J} d_j\) (a valid big‑\(M\)).  

3. **Variable domains**

\[
y_i\in\{0,1\}\quad \forall i\in I,\qquad
x_{ij}\ge 0\quad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Row/Column Identifier | Notes |
|-----------|--------------|-----------------------|-------|
| \(f_i\) | `file_1_view_0` | Row id column `"Unnamed: 0"` (supplier) | Column `"fixed_costs"` |
| \(d_j\) | `file_0_view_0` | Row id column `"customer"` (supermarket) | Column `"demand"` |
| \(c_{ij}\) | `file_2_view_0` | Row id column `"Unnamed: 0"` (supplier), Column id `"customer"` (supermarket) | Matrix entries |

All index sets \(I\) and \(J\) are derived from the distinct identifiers present in the respective source tables. No numerical values are enumerated in the model; they are bound to the data via the mapping above.