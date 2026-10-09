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
   x_{ij} \leq U_{ij} y_i, \quad \forall i \in I,\, j \in J
   \]
   where $U_{ij}$ is any valid upper bound on $x_{ij}$ (e.g., $U_{ij} = \sum_{j \in J} d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: set of branches, from `customer_id` in `file_0_view_0` and columns in `file_2_view_0`
- $d_j$: demand of branch $j$, from `demand_units` in `file_0_view_0`
- $f_i$: fixed opening cost for supplier $i$, from `fixed_opening_cost` in `file_1_view_0`
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from `transportation_cost_to_{j}` in `file_2_view_0`

##### Data Mapping

- $I$: All `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: All `customer_id` in `file_0_view_0` and suffixes of `transportation_cost_to_*` in `file_2_view_0`
- $d_j$: `file_0_view_0`, column `demand_units`, indexed by `customer_id`
- $f_i$: `file_1_view_0`, column `fixed_opening_cost`, indexed by `facility_id`
- $c_{ij}$: `file_2_view_0`, column `transportation_cost_to_{j}`, indexed by `facility_id` and $j$
- $x_{ij}$, $y_i$: decision variables as defined above

No supplier capacity limits are specified, so $U_{ij}$ can be set to $\sum_{j \in J} d_j$ for all $i,j$.

All index sets and parameters are defined by the full set of entities in the current CSV files.