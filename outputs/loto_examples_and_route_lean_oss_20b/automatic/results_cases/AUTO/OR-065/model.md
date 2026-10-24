## Mathematical Model

**Indices**  
\(I=\{S1,S2,S3\}\) (warehouses)  
\(J=\{C1,C2,C3\}\) (customers)

**Parameters**  
\(f_i\) fixed cost of warehouse \(i\) (from *fixed_cost.csv*)  
\(c_{ij}\) transportation cost per unit from warehouse \(i\) to customer \(j\) (from *transportation_costs.csv*)  
\(d_j\) demand of customer \(j\) (from *demand.csv*)  
\(U=\sum_{j\in J} d_j\) (upper bound for any shipment)

**Decision Variables**  
\(y_i\in\{0,1\}\) 1 if warehouse \(i\) is activated, 0 otherwise  
\(x_{ij}\ge 0\) quantity shipped from warehouse \(i\) to customer \(j\)

**Objective**  
\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
\]

2. **Warehouse activation**  
\[
x_{ij} \;\le\; U\,y_i \qquad \forall i\in I,\; j\in J
\]

3. **Variable domains**  
\[
y_i \in \{0,1\}\quad \forall i\in I,\qquad
x_{ij}\ge 0\quad \forall i\in I,\; j\in J
\]

---

## Data Mapping Rules

| Parameter | Source Table | Source Column | Index Mapping |
|-----------|--------------|---------------|---------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | `i` ← `Unnamed: 0` |
| \(c_{ij}\) | `file_2_view_0` | `Cj` (e.g., `C1`, `C2`, `C3`) | `i` ← `Unnamed: 0`, `j` ← column header |
| \(d_j\) | `file_0_view_0` | `demand` | `j` ← `customer` |
| \(U\) | computed | – | \(U = \sum_{j} d_j\) |

All indices \(I\) and \(J\) are taken from the full set of rows in the corresponding source tables; no additional filtering is applied. The model uses the exact values from the CSV files as parameters.