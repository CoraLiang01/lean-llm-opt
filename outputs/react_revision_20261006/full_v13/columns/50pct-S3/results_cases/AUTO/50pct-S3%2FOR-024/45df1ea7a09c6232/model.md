##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each musician/band $j$ receives exactly its demand.)

2. **Warehouse activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   (No goods can be shipped from warehouse $i$ unless it is activated. $M_i$ is a valid upper bound, e.g., $M_i = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of warehouses, from `file_1_view_0` column `Unnamed: 0`
- $J$: Set of musicians/bands, from `file_0_view_0` column `customer`
- $d_j$: Demand of musician/band $j$, from `file_0_view_0` column `demand`
- $f_i$: Fixed cost for warehouse $i$, from `file_1_view_0` column `fixed_costs`
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$, from `file_2_view_0` (row: `Unnamed: 0` = $i$, column: $j$)
- $M_i$: Big-M upper bound for each warehouse $i$, set as $M_i = \sum_{j \in J} d_j$ (sum of all demands, from `file_0_view_0` column `demand`)

##### Data Mapping

- Warehouses $I$: `file_1_view_0` column `Unnamed: 0`
- Musicians/Bands $J$: `file_0_view_0` column `customer`
- Demand $d_j$: `file_0_view_0` columns `customer`, `demand`
- Fixed cost $f_i$: `file_1_view_0` columns `Unnamed: 0`, `fixed_costs`
- Transportation cost $c_{ij}$: `file_2_view_0` (row: `Unnamed: 0` = $i$, column: $j$)
- $M_i$: $\sum_{j \in J} d_j$ (from `file_0_view_0` column `demand`)

No additional capacity or proportion constraints are imposed beyond those above. All variables and parameters are mapped directly to the provided CSV data.