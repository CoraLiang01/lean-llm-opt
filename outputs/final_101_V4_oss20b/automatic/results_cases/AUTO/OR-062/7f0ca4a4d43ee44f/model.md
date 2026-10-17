**Mathematical Model**

Let  

* \(I\) – set of suppliers (rows of *fixed_cost.csv*)  
* \(J\) – set of stores (rows of *demand.csv*)  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_i\) | Fixed cost of opening supplier \(i\) | `file_1_view_0` column **fixed_costs** |
| \(d_j\) | Demand of store \(j\) | `file_0_view_0` column **demand** |
| \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to store \(j\) | `file_2_view_0` – row \(i\), column \(j\) |
| \(M\) | Big‑\(M\) bound (e.g. \(\sum_{j\in J} d_j\)) | derived from \(d_j\) |

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij} \ge 0\) – quantity shipped from supplier \(i\) to store \(j\)

Objective  

\[
\min \;\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} = d_j , \qquad \forall j\in J
   \]

2. **Supplier activation (big‑\(M\) linking)**  
   \[
   \sum_{j\in J} x_{ij} \;\le\; M\,y_i , \qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}, \qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row identifier is the value in column **Unnamed: 0** |
| \(d_j\) | `file_0_view_0` | `demand` | Row identifier is the value in column **Customer** |
| \(c_{ij}\) | `file_2_view_0` | Columns **CLARINDA**, **FORT MADISON**, **SIOUX CITY**, **TOLEDO**, **BANCROFT** | Row identifier is the value in column **Unnamed: 0** (supplier) |
| \(M\) | derived | – | \(M = \sum_{j\in J} d_j\) |

All indices \(i\) and \(j\) are taken from the full sets of suppliers and stores present in the respective CSV files. No additional capacity limits are imposed on suppliers.