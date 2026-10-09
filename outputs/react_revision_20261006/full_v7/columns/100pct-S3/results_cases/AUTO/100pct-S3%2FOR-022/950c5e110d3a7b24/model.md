##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

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
   x_{ij} \leq U_{ij} y_i, \quad \forall i \in I, \forall j \in J
   \]
   where $U_{ij}$ is any valid upper bound on $x_{ij}$ (e.g., $U_{ij} = d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from `facility_id` in `file_1_view_0` and `file_2_view_0`.
- $J$: Set of branches, from `customer_id` in `file_0_view_0` and columns in `file_2_view_0`.
- $d_j$: Demand of branch $j$, from `demand_units` in `file_0_view_0`.
- $f_i$: Fixed opening cost for supplier $i$, from `fixed_opening_cost` in `file_1_view_0`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$, from `transportation_cost_to_{j}` in `file_2_view_0`.

##### Data Mapping

- $I$: All `facility_id` in `file_1_view_0` and `file_2_view_0`.
- $J$: All `customer_id` in `file_0_view_0` and suffixes of `transportation_cost_to_{j}` columns in `file_2_view_0`.
- $d_j$: `demand_units` column in `file_0_view_0`, indexed by `customer_id`.
- $f_i$: `fixed_opening_cost` column in `file_1_view_0`, indexed by `facility_id`.
- $c_{ij}$: `transportation_cost_to_{j}` columns in `file_2_view_0`, indexed by `facility_id` and branch $j$.
- $x_{ij}$, $y_i$: Decision variables as defined above.

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files. No capacity or supply bounds are imposed beyond those implied by demand and activation.