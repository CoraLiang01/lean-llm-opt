**Mathematical Model**

Let  

- \(I=\{S1,S2,S3,S4,S5\}\) (suppliers)  
- \(J=\{C1,C2,C3,C4,C5\}\) (branches)

**Parameters**

| Symbol | Meaning | Data source |
|--------|---------|-------------|
| \(d_j\) | Demand of branch \(j\) | `file_0_view_0` column **demand** |
| \(f_i\) | Fixed cost of opening supplier \(i\) | `file_1_view_0` column **fixed_costs** |
| \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to branch \(j\) | `file_2_view_0` columns **C1**–**C5** |
| \(M\) | Big‑\(M\) upper bound on total shipment from a supplier | \(M=\sum_{j\in J} d_j\) (computed from \(d_j\)) |

**Decision Variables**

- \(x_{ij}\ge 0\) quantity shipped from supplier \(i\) to branch \(j\)  
- \(y_i\in\{0,1\}\) indicator that supplier \(i\) is opened

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Supplier activation (big‑\(M\) constraint)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i ,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\},\qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(d_j\) | `file_0_view_0` | `demand` | Row key: `customer` |
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row key: `Unnamed: 0` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C5` | Row key: `Unnamed: 0`; column key: `customer` |
| \(M\) | computed | – | Sum of all \(d_j\) values |

All indices and parameters are defined solely from the data tables provided; no additional limits or capacities are imposed beyond the big‑\(M\) bound.