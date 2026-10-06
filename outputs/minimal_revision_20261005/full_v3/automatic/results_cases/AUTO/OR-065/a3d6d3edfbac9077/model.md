##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i$ to musician/band $j$ (continuous), for all warehouses $i$ and musicians/bands $j$.
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each musician/band $j$ receives exactly their demand.)

2. **Warehouse activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (No goods can be shipped from inactive warehouses. $M = \sum_{j \in J} d_j$ is a valid upper bound.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of warehouses (from `file_1_view_0`, column `Unnamed: 0`)
- $J$: Set of musicians/bands (from `file_0_view_0`, column `customer`)

##### Parameters and Data Mapping

- $d_j$: Demand of musician/band $j$  
  — Source: `file_0_view_0`, columns: `customer`, `demand`
- $f_i$: Fixed cost for warehouse $i$  
  — Source: `file_1_view_0`, columns: `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$  
  — Source: `file_2_view_0`, rows: `Unnamed: 0` (warehouse), columns: `C1`, `C2`, `C3` (musician/band)
- $M = \sum_{j \in J} d_j$ (total demand, computed from `file_0_view_0`)

##### Data Mapping

- Warehouses $I$:  
  — `file_1_view_0`, column `Unnamed: 0`  
  — Row IDs: S1, S2, S3

- Musicians/Bands $J$:  
  — `file_0_view_0`, column `customer`  
  — Column IDs: C1, C2, C3

- Demand $d_j$:  
  — `file_0_view_0`, columns: `customer`, `demand`

- Fixed cost $f_i$:  
  — `file_1_view_0`, columns: `Unnamed: 0`, `fixed_costs`

- Transportation cost $c_{ij}$:  
  — `file_2_view_0`, row_id_mapping: S1, S2, S3; column_id_mapping: C1, C2, C3

- $M$:  
  — $M = \sum_{j \in J} d_j$, with $d_j$ from `file_0_view_0`, column `demand`