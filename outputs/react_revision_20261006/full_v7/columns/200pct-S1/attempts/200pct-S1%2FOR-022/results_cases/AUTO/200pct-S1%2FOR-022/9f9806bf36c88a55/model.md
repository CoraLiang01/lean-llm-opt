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
   x_{ij} \leq D_j y_i, \quad \forall i \in I, \forall j \in J
   \]
   where $D_j$ is the demand of branch $j$ (ensures $x_{ij}=0$ if $y_i=0$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (facility IDs from `fixed_cost.csv` and `transportation_costs.csv`)
- $J$: Set of branches/customers (customer IDs from `demand.csv`)
- $d_j$: Demand of branch $j$ (from `demand.csv`)
- $f_i$: Fixed opening cost for supplier $i$ (from `fixed_cost.csv`)
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to branch $j$ (from `transportation_costs.csv`)

##### Data Mapping

- $I$: All `facility_id` in `file_1_view_0` and `file_2_view_0`
- $J$: All `customer_id` in `file_0_view_0`
- $d_j$: `demand_units` column in `file_0_view_0`, indexed by `customer_id`
- $f_i$: `fixed_opening_cost` column in `file_1_view_0`, indexed by `facility_id`
- $c_{ij}$: `transportation_cost_to_Ck` columns in `file_2_view_0`, indexed by `facility_id` (rows) and customer $j$ (columns, where $k$ matches $j$)
- $x_{ij}, y_i$: Decision variables as defined above

No supplier capacity limits are specified; only activation and demand constraints apply.