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
   (Each musician/band $j$ receives exactly its demand $d_j$.)

2. **Warehouse activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   (No goods can be shipped from warehouse $i$ unless it is activated. $M_i$ is a sufficiently large upper bound, e.g., $M_i = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets and Parameters

- $I$: Set of warehouses, from column "Unnamed: 0" in `fixed_cost.csv` and `transportation_costs.csv`.
- $J$: Set of musicians/bands, from column "customer" in `demand.csv` and columns in `transportation_costs.csv` (excluding "Unnamed: 0").
- $d_j$: Demand of musician/band $j$, from column "demand" in `demand.csv`.
- $f_i$: Fixed cost for warehouse $i$, from column "fixed_costs" in `fixed_cost.csv`.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$, from `transportation_costs.csv` (row "Unnamed: 0" = $i$, column $j$).
- $M_i$: Big-M parameter for each warehouse $i$, set as $M_i = \sum_{j \in J} d_j$.

##### Data Mapping

- $I$: All values in column "Unnamed: 0" of `fixed_cost.csv` (table_id: file_1_view_0) and `transportation_costs.csv` (table_id: file_2_view_0).
- $J$: All values in column "customer" of `demand.csv` (table_id: file_0_view_0) and columns (excluding "Unnamed: 0") of `transportation_costs.csv` (table_id: file_2_view_0).
- $d_j$: Column "demand" in `demand.csv` (table_id: file_0_view_0), indexed by "customer".
- $f_i$: Column "fixed_costs" in `fixed_cost.csv` (table_id: file_1_view_0), indexed by "Unnamed: 0".
- $c_{ij}$: Entry in `transportation_costs.csv` (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column $j$.
- $M_i$: $M_i = \sum_{j \in J} d_j$, where $d_j$ as above.

No additional constraints or capacity limits are specified beyond those above.