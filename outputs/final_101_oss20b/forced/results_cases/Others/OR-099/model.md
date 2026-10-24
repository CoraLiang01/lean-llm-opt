## Mathematical Model

**Index sets**  
\(I=\{1,\dots ,11\}\) – warehouses  
\(J=\{1,\dots ,11\}\) – stores  

**Parameters**  
\(f_i\) – opening cost of warehouse \(i\) (from *PotentialWarehouses_Costs.csv*, column *Opening Cost (fi)*)  
\(C_i\) – capacity of warehouse \(i\) (from *PotentialWarehouses_Costs.csv*, column *Capacity (units)*)  
\(d_j\) – demand of store \(j\) (from *Stores_Demands.csv*, column *Demand (units, dj)*)  
\(c_{ij}\) – transportation cost per unit from warehouse \(i\) to store \(j\) (from *TransportationCost.csv*, matrix entry \((j,i)\))

**Decision variables**  
\(y_i\in\{0,1\}\) = 1 if warehouse \(i\) is opened, 0 otherwise  
\(x_{ij}\ge 0\) = units shipped from warehouse \(i\) to store \(j\)

**Objective**  
\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J
\]

2. **Warehouse capacity**  
\[
\sum_{j\in J} x_{ij}\;\le\; C_i\,y_i \qquad \forall i\in I
\]

3. **Non‑negativity**  
\[
x_{ij}\ge 0 \qquad \forall i\in I,\; j\in J
\]

4. **Binary opening decisions**  
\[
y_i\in\{0,1\} \qquad \forall i\in I
\]

---

## Data Mapping

| Parameter | Source Table | Column / Matrix Entry | Mapping Notes |
|-----------|--------------|-----------------------|---------------|
| \(f_i\) | `file_0_view_0` | `Opening Cost (fi)` | One value per warehouse \(i\) |
| \(C_i\) | `file_0_view_0` | `Capacity (units)` | One value per warehouse \(i\) |
| \(d_j\) | `file_1_view_0` | `Demand (units, dj)` | One value per store \(j\) |
| \(c_{ij}\) | `file_2_view_0` | Matrix entry at row *Store (j)*, column *Warehouse (i)* | 11 × 11 matrix, row and column indices aligned to `Warehouse (i)` and `Store (j)` |

*Index sets \(I\) and \(J\) are derived from the distinct values of `Warehouse (i)` and `Store (j)` in the respective tables.*