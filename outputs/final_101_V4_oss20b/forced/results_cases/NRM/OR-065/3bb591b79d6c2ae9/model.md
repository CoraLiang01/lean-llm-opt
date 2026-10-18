##### Sets
- \(I\): set of warehouses  
- \(J\): set of customers  

##### Parameters
- \(f_i\): fixed cost of operating warehouse \(i\)  (\(i \in I\))  
- \(d_j\): demand of customer \(j\)  (\(j \in J\))  
- \(c_{ij}\): transportation cost per unit from warehouse \(i\) to customer \(j\)  (\(i \in I,\; j \in J\))  

##### Variables
- \(y_i \in \{0,1\}\)  (warehouse \(i\) is activated)  
- \(x_{ij} \ge 0\)  (quantity supplied from warehouse \(i\) to customer \(j\))  

##### Objective
\[
\min \;\sum_{i \in I} f_i\,y_i \;+\; \sum_{i \in I}\sum_{j \in J} c_{ij}\,x_{ij}
\]

##### Constraints  

1. **Demand satisfaction**  
\[
\sum_{i \in I} x_{ij} \;=\; d_j \quad \forall j \in J
\]

2. **Supply only from active warehouses**  
\[
x_{ij} \;\le\; d_j\,y_i \quad \forall i \in I,\; j \in J
\]

3. **Binary activation**  
\[
y_i \in \{0,1\} \quad \forall i \in I
\]

4. **Non‑negativity**  
\[
x_{ij} \;\ge\; 0 \quad \forall i \in I,\; j \in J
\]

##### Data Mapping  

| Source | Table ID | Column Names |
|--------|----------|--------------|
| Fixed cost data | `file_1_view_0` | `Unnamed: 0` (warehouse id), `fixed_costs` |
| Demand data | `file_0_view_0` | `customer`, `demand` |
| Transportation costs | `file_2_view_0` | `Unnamed: 0` (warehouse id), `C1`, `C2`, `C3` |

*The column `Unnamed: 0` in each table represents the warehouse identifier.*