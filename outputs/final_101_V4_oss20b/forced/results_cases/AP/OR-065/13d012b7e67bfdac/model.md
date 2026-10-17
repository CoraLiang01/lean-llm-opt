**Mathematical Model**

Let  

* \(I=\{S1,S2,S3\}\) – set of distribution warehouses  
* \(J=\{C1,C2,C3\}\) – set of musicians/bands (customers)

Parameters  

* \(f_i\) – fixed cost of opening warehouse \(i\)  
* \(c_{ij}\) – transportation cost per unit from warehouse \(i\) to customer \(j\)  
* \(d_j\) – demand of customer \(j\)

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if warehouse \(i\) is activated, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from warehouse \(i\) to customer \(j\)

---

### Objective
\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

### Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;=\; d_j \qquad \forall j\in J
   \]

2. **Warehouse activation**  
   \[
   x_{ij} \;\le\; U\,y_i \qquad \forall i\in I,\; j\in J
   \]
   where \(U=\sum_{j\in J} d_j\) (a sufficiently large constant).

3. **Variable domains**  
   \[
   y_i \in \{0,1\}\quad \forall i\in I,\qquad
   x_{ij}\ge 0\quad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Notes |
|-----------|--------------|------------|--------------|-------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | Warehouse identifier |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` (row) | `C1`, `C2`, `C3` (columns) | Transportation cost matrix |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | Customer demand |

All indices \(i\) and \(j\) are taken from the respective key columns of the source tables. No additional data are required.