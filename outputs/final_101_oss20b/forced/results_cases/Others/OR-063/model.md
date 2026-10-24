## Mathematical Model

**Sets**  
\(I=\{S_1,S_2,\dots ,S_7\}\) (warehouses)  
\(J=\{C_1,C_2,\dots ,C_7\}\) (customers)

**Parameters**  
\(f_i\) fixed cost of opening warehouse \(i\) (from *fixed_cost.csv*)  
\(t_{ij}\) transportation cost per unit from warehouse \(i\) to customer \(j\) (from *transportation_costs.csv*)  
\(d_j\) demand of customer \(j\) (from *demand.csv*)

**Decision Variables**  
\(y_i\in\{0,1\}\) 1 if warehouse \(i\) is activated, 0 otherwise  
\(x_{ij}\ge 0\) quantity supplied from warehouse \(i\) to customer \(j\)

**Objective**  
\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
\]

2. **Linking supply to activation**  
\[
x_{ij} \;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J
\]

3. **Variable domains**  
\[
y_i\in\{0,1\}\;\;\forall i\in I,\qquad
x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Key Column | Value Column | Mapping Rule |
|-----------|--------------|------------|--------------|--------------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | \(f_i =\) value where `Unnamed: 0` = \(i\) |
| \(t_{ij}\) | `file_2_view_0` | `Unnamed: 0` (row) | `C1`–`C7` (columns) | \(t_{ij} =\) cell at row `Unnamed: 0` = \(i\), column = \(j\) |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | \(d_j =\) value where `customer` = \(j\) |

All indices \(i\) and \(j\) are taken from the full set of rows/columns present in the respective source tables, as returned in the observation. No additional filtering or aggregation is applied.