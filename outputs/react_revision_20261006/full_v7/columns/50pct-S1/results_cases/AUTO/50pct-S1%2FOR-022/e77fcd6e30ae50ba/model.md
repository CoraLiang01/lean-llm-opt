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
   where $D_j$ is the demand of branch $j$ (ensures $x_{ij}=0$ if $y_i=0$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers (facility IDs from `fixed_cost.csv` and `transportation_costs.csv`)
- $J$: set of branches/customers (customer IDs from `demand.csv` and columns of `transportation_costs.csv`)
- $d_j$: demand of branch $j$ (from `demand.csv`)
- $f_i$: fixed opening cost for supplier $i$ (from `fixed_cost.csv`)
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$ (from `transportation_costs.csv`)

##### Data Mapping

- $I$: All `facility_id` in table_id `file_1_view_0` and `file_2_view_0`
- $J$: All `customer_id` in table_id `file_0_view_0` and columns `transportation_cost_to_*` in table_id `file_2_view_0`
- $d_j$: `demand_units` column in table_id `file_0_view_0`, indexed by `customer_id`
- $f_i$: `fixed_opening_cost` column in table_id `file_1_view_0`, indexed by `facility_id`
- $c_{ij}$: `transportation_cost_to_*` columns in table_id `file_2_view_0`, indexed by `facility_id` (rows) and customer suffix (columns)
- $D_j$: $d_j$ as above

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files. No values are omitted or abbreviated.