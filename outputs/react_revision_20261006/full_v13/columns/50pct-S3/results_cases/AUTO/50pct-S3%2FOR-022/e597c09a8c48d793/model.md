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
   x_{ij} \leq D_j y_i, \quad \forall i \in I, \forall j \in J
   \]
   where $D_j$ is any valid upper bound on demand for branch $j$ (e.g., $D_j = d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: set of branches, from `customer_id` in `file_0_view_0` and columns in `file_2_view_0` (after mapping)
- $d_j$: demand for branch $j$, from `demand_units` in `file_0_view_0`
- $f_i$: fixed opening cost for supplier $i$, from `fixed_opening_cost` in `file_1_view_0`
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from `transportation_cost_to_*` columns in `file_2_view_0`

##### Data Mapping

- $I$: All `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: All `customer_id` in `file_0_view_0` and mapped from `transportation_cost_to_*` columns in `file_2_view_0`
- $d_j$: `file_0_view_0`, column `demand_units`, indexed by `customer_id`
- $f_i$: `file_1_view_0`, column `fixed_opening_cost`, indexed by `facility_id`
- $c_{ij}$: `file_2_view_0`, columns `transportation_cost_to_C1`, ..., `transportation_cost_to_C5`, indexed by `facility_id` and mapped branch $j$
- $x_{ij}$, $y_i$: decision variables as defined above

All index sets and parameters are defined by the full set of entities in the current CSV files. No capacity limits are imposed except those implied by demand satisfaction and activation logic.