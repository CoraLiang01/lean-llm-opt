## Mathematical Model

**Sets**  
\(I=\{S1,S2,S3\}\) (warehouses)  
\(J=\{C1,C2,C3\}\) (customers)

**Parameters**  
\(f_i\) fixed cost of warehouse \(i\) (from *fixed_cost.csv*)  
\(c_{ij}\) transportation cost per unit from warehouse \(i\) to customer \(j\) (from *transportation_costs.csv*)  
\(d_j\) demand of customer \(j\) (from *demand.csv*)

**Decision variables**  
\(y_i\in\{0,1\}\) warehouse \(i\) is activated  
\(x_{ij}\ge 0\) quantity supplied from warehouse \(i\) to customer \(j\)

**Objective**  
\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
\]

2. **Linking warehouse activation to supply**  
\[
x_{ij} \;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J
\]

3. **Variable domains**  
\[
y_i \in \{0,1\}\;\;\forall i\in I,\qquad
x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Row id `Unnamed: 0` gives warehouse id |
| \(c_{ij}\) | `file_2_view_0` | `C1`, `C2`, `C3` | Row id `Unnamed: 0` gives warehouse id; column names give customer id |
| \(d_j\) | `file_0_view_0` | `demand` | Row id `customer` gives customer id |

All indices \(i\) and \(j\) are taken from the complete set of rows in the respective tables; no additional filtering is applied. The model uses the exact values as read from the CSV files.