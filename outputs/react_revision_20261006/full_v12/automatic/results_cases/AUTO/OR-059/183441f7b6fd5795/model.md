##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Dealership demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ is a valid upper bound for each supplier (since no explicit supplier capacity is given).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` and `transportation_costs.csv`.
- $J$: Set of dealerships, from column "customer" in `demand.csv` and columns in `transportation_costs.csv` (excluding "Unnamed: 0").
- $d_j$: Demand of dealership $j$, from column "demand" in `demand.csv`.
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in `fixed_cost.csv`.
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from `transportation_costs.csv`.

##### Data Mapping

- $I$: All values in "Unnamed: 0" of `fixed_cost.csv` and `transportation_costs.csv` (table_id: file_1_view_0, file_2_view_0)
- $J$: All values in "customer" of `demand.csv` and columns (except "Unnamed: 0") in `transportation_costs.csv` (table_id: file_0_view_0, file_2_view_0)
- $d_j$: "demand" column in `demand.csv` (table_id: file_0_view_0)
- $f_i$: "fixed_costs" column in `fixed_cost.csv` (table_id: file_1_view_0)
- $c_{ij}$: Matrix entries in `transportation_costs.csv` (table_id: file_2_view_0, rows: suppliers, columns: dealerships)
- $M_i$: $\sum_{j \in J} d_j$ (sum over all "demand" in `demand.csv`)

All indices and parameters are defined by the full set of current records in the respective CSV files.