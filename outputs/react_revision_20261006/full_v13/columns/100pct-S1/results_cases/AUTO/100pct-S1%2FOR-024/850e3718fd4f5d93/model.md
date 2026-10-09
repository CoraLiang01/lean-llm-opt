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

2. **Warehouse activation:**  
   \[
   x_{ij} \leq d_j y_i, \quad \forall i \in I,\, j \in J
   \]
   (A warehouse can only supply to a musician/band if it is activated.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of warehouses, from `file_1_view_0` column `Unnamed: 0` and `file_2_view_0` row `Unnamed: 0` (IDs: S1, S2, S3).
- $J$: Set of musicians/bands, from `file_0_view_0` column `customer` and `file_2_view_0` columns (IDs: C1, C2, C3).
- $d_j$: Demand of musician/band $j$, from `file_0_view_0` columns `customer`, `demand`.
- $f_i$: Fixed cost for warehouse $i$, from `file_1_view_0` columns `Unnamed: 0`, `fixed_costs`.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$, from `file_2_view_0` rows `Unnamed: 0`, columns `C1`, `C2`, `C3`.

##### Data Mapping

- $I$: `file_1_view_0` column `Unnamed: 0` and `file_2_view_0` row `Unnamed: 0`
- $J$: `file_0_view_0` column `customer` and `file_2_view_0` columns `C1`, `C2`, `C3`
- $d_j$: `file_0_view_0` columns `customer`, `demand`
- $f_i$: `file_1_view_0` columns `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: `file_2_view_0` rows `Unnamed: 0`, columns `C1`, `C2`, `C3`