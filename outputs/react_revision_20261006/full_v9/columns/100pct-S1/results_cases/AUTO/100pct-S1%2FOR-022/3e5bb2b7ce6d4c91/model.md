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
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: set of branches, from `customer_id` in `file_0_view_0`
- $d_j$: demand of branch $j$, from `demand_units` in `file_0_view_0`
- $f_i$: fixed opening cost for supplier $i$, from `fixed_opening_cost` in `file_1_view_0`
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from `transportation_cost_to_{j}` in `file_2_view_0`
- $M_i$: big-M upper bound for supplier $i$, set to $\sum_{j \in J} d_j$

##### Data Mapping

- $I$: All `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: All `customer_id` in `file_0_view_0`
- $d_j$: `demand_units` column, `file_0_view_0`, indexed by `customer_id`
- $f_i$: `fixed_opening_cost` column, `file_1_view_0`, indexed by `facility_id`
- $c_{ij}$: `transportation_cost_to_{j}` columns, `file_2_view_0`, indexed by `facility_id` and mapped to $j$ via the column-axis mapping in the Observation
- $M_i$: $\sum_{j \in J} d_j$, with $d_j$ as above

- Matrix axes:  
  - Rows: `facility_id` from `file_1_view_0` and `file_2_view_0`  
  - Columns: `customer_id` from `file_0_view_0` (via mapping in Observation)

All index sets and parameters are defined by the full set of entities in the current CSV files. No supplier capacity constraints are imposed beyond the activation logic.