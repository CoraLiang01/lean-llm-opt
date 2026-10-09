##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Dealership demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each dealership $j$ receives exactly its demand $d_j$.)

2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (Supplier $i$ can only ship if open; $M$ is a sufficiently large upper bound, e.g., $M = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` and `transportation_costs.csv` (table_id: file_1_view_0, file_2_view_0).
- $J$: Set of dealerships, from column "customer" in `demand.csv` and columns in `transportation_costs.csv` (table_id: file_0_view_0, file_2_view_0).
- $d_j$: Demand of dealership $j$, from column "demand" in `demand.csv` (table_id: file_0_view_0).
- $f_i$: Fixed cost for opening supplier $i$, from column "fixed_costs" in `fixed_cost.csv` (table_id: file_1_view_0).
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from `transportation_costs.csv` (table_id: file_2_view_0, row "Unnamed: 0" for $i$, column $j$).
- $M$: $M = \sum_{j \in J} d_j$ (sum of all dealership demands).

##### Data Mapping

- $I$: All values in "Unnamed: 0" of `fixed_cost.csv` (table_id: file_1_view_0) and rows of `transportation_costs.csv` (table_id: file_2_view_0).
- $J$: All values in "customer" of `demand.csv` (table_id: file_0_view_0) and columns (except "Unnamed: 0") of `transportation_costs.csv` (table_id: file_2_view_0).
- $d_j$: "demand" column in `demand.csv` (table_id: file_0_view_0), indexed by "customer".
- $f_i$: "fixed_costs" column in `fixed_cost.csv` (table_id: file_1_view_0), indexed by "Unnamed: 0".
- $c_{ij}$: Entry in `transportation_costs.csv` (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column $j$.
- $M$: $\sum_{j \in J} d_j$, with $d_j$ as above.

All index sets and parameters are defined by the full set of current records in the respective CSV files.