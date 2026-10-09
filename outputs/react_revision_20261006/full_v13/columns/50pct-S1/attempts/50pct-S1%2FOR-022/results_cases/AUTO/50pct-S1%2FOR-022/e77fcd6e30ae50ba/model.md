##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction:**  
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

##### Index Sets

- $I$: set of suppliers, from column `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: set of branches, from column `customer_id` in `file_0_view_0`

##### Parameters and Data Mapping

- $d_j$: demand of branch $j$, from column `demand_units` in `file_0_view_0` (table_id: file_0_view_0, key: customer_id)
- $f_i$: fixed opening cost for supplier $i$, from column `fixed_opening_cost` in `file_1_view_0` (table_id: file_1_view_0, key: facility_id)
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from column `transportation_cost_to_{j}` in `file_2_view_0` (table_id: file_2_view_0, row: facility_id, column: transportation_cost_to_{customer_id})
- $M_i$: big-M upper bound for supplier $i$, set as $\sum_{j \in J} d_j$ (sum over all $d_j$ from file_0_view_0)

##### Data Mapping

- $I$: All `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: All `customer_id` in `file_0_view_0`
- $d_j$: `file_0_view_0`, columns: `customer_id`, `demand_units`
- $f_i$: `file_1_view_0`, columns: `facility_id`, `fixed_opening_cost`
- $c_{ij}$: `file_2_view_0`, row: `facility_id`, column: `transportation_cost_to_{customer_id}`
- $M_i$: $\sum_{j \in J} d_j$ (sum of all `demand_units` in `file_0_view_0`)

No supplier capacity constraints are present except those implied by activation. All variables and parameters are mapped directly to the CSV data as described.