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
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (No goods can be shipped from inactive warehouses. $M = \sum_{j \in J} d_j$ is a valid upper bound.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of warehouses, from column `Unnamed: 0` in `file_1_view_0` (fixed_cost.csv) and `file_2_view_0` (transportation_costs.csv).
- $J$: Set of musicians/bands, from column `customer` in `file_0_view_0` (demand.csv) and columns in `file_2_view_0` (transportation_costs.csv).
- $d_j$: Demand for musician/band $j$, from column `demand` in `file_0_view_0` (demand.csv).
- $f_i$: Fixed cost for warehouse $i$, from column `fixed_costs` in `file_1_view_0` (fixed_cost.csv).
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$, from matrix in `file_2_view_0` (transportation_costs.csv), with rows indexed by `Unnamed: 0` (warehouse) and columns by musician/band IDs.
- $M = \sum_{j \in J} d_j$: Total demand, computed from all $d_j$.

##### Data Mapping

- $I$: All values in `Unnamed: 0` of `file_1_view_0` and `file_2_view_0`
- $J$: All values in `customer` of `file_0_view_0` and columns (except `Unnamed: 0`) of `file_2_view_0`
- $d_j$: `demand` column in `file_0_view_0`, keyed by `customer`
- $f_i$: `fixed_costs` column in `file_1_view_0`, keyed by `Unnamed: 0`
- $c_{ij}$: Matrix in `file_2_view_0`, rows by `Unnamed: 0`, columns by musician/band IDs
- $M$: $\sum_{j \in J} d_j$ (sum over all `demand` in `file_0_view_0`)

No additional constraints or capacity limits are specified. All variables and parameters are mapped directly to the provided CSV data.