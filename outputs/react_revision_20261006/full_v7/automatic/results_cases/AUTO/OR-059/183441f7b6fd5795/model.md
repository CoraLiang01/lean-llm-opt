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
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand), a valid upper bound since there are no explicit supplier capacity limits.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` and row "Unnamed: 0" in `transportation_costs.csv`.
- $J$: Set of dealerships, from column "customer" in `demand.csv` and columns "C1", ..., "C9" in `transportation_costs.csv`.
- $d_j$: Demand of dealership $j$, from column "demand" in `demand.csv`.
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in `fixed_cost.csv`.
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from `transportation_costs.csv` (row "Unnamed: 0" = $i$, column $j$).
- $M = \sum_{j \in J} d_j$, computed from all $d_j$ in `demand.csv`.

##### Data Mapping

- $I$: All values in column "Unnamed: 0" of `fixed_cost.csv` (table_id: file_1_view_0) and row "Unnamed: 0" of `transportation_costs.csv` (table_id: file_2_view_0).
- $J$: All values in column "customer" of `demand.csv` (table_id: file_0_view_0) and columns "C1"–"C9" of `transportation_costs.csv` (table_id: file_2_view_0).
- $d_j$: Column "demand" in `demand.csv` (table_id: file_0_view_0), indexed by "customer".
- $f_i$: Column "fixed_costs" in `fixed_cost.csv` (table_id: file_1_view_0), indexed by "Unnamed: 0".
- $c_{ij}$: Entry in `transportation_costs.csv` (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column $j$.
- $M$: $\sum_{j \in J} d_j$, with $d_j$ as above.

No supplier capacity limits are present; $M$ is used as a big-M upper bound for activation logic. All indices and parameters are defined by the full set of current CSV records.