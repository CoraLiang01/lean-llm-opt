**Mathematical Model**

Let  

- \(I=\{S_1,S_2,\dots ,S_7\}\) (warehouses)  
- \(J=\{C_1,C_2,\dots ,C_7\}\) (customers)  

Parameters  

- \(d_j\) : demand of customer \(j\)  \(d_j =\) value in *demand.csv* where column = \(j\).  
- \(f_i\) : fixed cost of warehouse \(i\) \(f_i =\) value in *fixed_cost.csv* where row = \(i\).  
- \(c_{ij}\) : unit transportation cost from warehouse \(i\) to customer \(j\) \(c_{ij} =\) value in *transportation_costs.csv* at row \(i\), column \(j\).  
- \(M = \sum_{j\in J} d_j\) (a big‑\(M\) bound, equal to total demand).  

Decision variables  

- \(x_{ij}\ge 0\) quantity shipped from warehouse \(i\) to customer \(j\).  
- \(y_i\in\{0,1\}\) indicator that warehouse \(i\) is activated.  

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i
\]

Constraints  

1. **Customer demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j,\qquad \forall j\in J
   \]

2. **Warehouse activation (big‑\(M\) constraint)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\},\qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping Rules**

| Parameter | Source File | Source Column(s) | Mapping |
|-----------|-------------|------------------|---------|
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv` | `customer` (row identifier), `C1`–`C7` (columns) | `d_j` = value in row where `customer` = \(j\), column = \(j\) |
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv` | `Unnamed: 0` (row identifier), `fixed_costs` (column) | `f_i` = value in row where `Unnamed: 0` = \(i\) |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv` | `Unnamed: 0` (row identifier), `C1`–`C7` (columns) | `c_{ij}` = value in row where `Unnamed: 0` = \(i\), column = \(j\) |
| \(M\) | Computed | – | \(M = \sum_{j\in J} d_j\) |

All identifiers (`S1`–`S7`, `C1`–`C7`) are taken verbatim from the respective files. No additional capacity limits are imposed beyond the big‑\(M\) bound.