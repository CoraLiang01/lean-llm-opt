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
   where $M_i = \sum_{j \in J} d_j$ is a sufficiently large upper bound for each supplier (since no explicit supplier capacity is given).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` and `transportation_costs.csv` (table_id: file_1_view_0, file_2_view_0).
- $J$: Set of dealerships, from column "customer" in `demand.csv` and columns in `transportation_costs.csv` (table_id: file_0_view_0, file_2_view_0).
- $d_j$: Demand of dealership $j$, from column "demand" in `demand.csv` (table_id: file_0_view_0).
- $f_i$: Fixed cost for opening supplier $i$, from column "fixed_costs" in `fixed_cost.csv` (table_id: file_1_view_0).
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from `transportation_costs.csv` (table_id: file_2_view_0, row "Unnamed: 0" for suppliers, columns for dealerships).
- $M_i$: Big-M parameter for each supplier $i$, set to $\sum_{j \in J} d_j$.

##### Data Mapping

- $I$: All values in "Unnamed: 0" from `fixed_cost.csv` (table_id: file_1_view_0) and `transportation_costs.csv` (table_id: file_2_view_0).
- $J$: All values in "customer" from `demand.csv` (table_id: file_0_view_0) and columns (except "Unnamed: 0") in `transportation_costs.csv` (table_id: file_2_view_0).
- $d_j$: "demand" column in `demand.csv` (table_id: file_0_view_0), keyed by "customer".
- $f_i$: "fixed_costs" column in `fixed_cost.csv` (table_id: file_1_view_0), keyed by "Unnamed: 0".
- $c_{ij}$: Entry in `transportation_costs.csv` (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column = $j$.
- $M_i$: $\sum_{j \in J} d_j$ (sum over all "demand" in `demand.csv`).

This model ensures all dealership demands are met, suppliers are only used if opened, and total cost (fixed + transportation) is minimized.