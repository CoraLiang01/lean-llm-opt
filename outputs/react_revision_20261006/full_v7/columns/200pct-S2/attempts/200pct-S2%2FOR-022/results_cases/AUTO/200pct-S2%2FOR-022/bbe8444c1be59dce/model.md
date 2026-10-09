##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier (facility) $i \in I$ to branch (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier (facility) $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ (since no explicit supplier capacity is given).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers (facilities), from `fixed_cost.csv` column `facility_id`.
- $J$: set of branches (customers), from `demand.csv` column `customer_id`.
- $d_j$: demand of branch $j$, from `demand.csv` column `demand_units`.
- $f_i$: fixed opening cost for supplier $i$, from `fixed_cost.csv` column `fixed_opening_cost`.
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from `transportation_costs.csv` column `transportation_cost_to_{j}` for each $j$.
- $M_i$: big-M upper bound for supplier $i$, set to $\sum_{j \in J} d_j$.

##### Data Mapping

- $I$: All `facility_id` in table_id `file_1_view_0` (`fixed_cost.csv`).
- $J$: All `customer_id` in table_id `file_0_view_0` (`demand.csv`).
- $d_j$: `demand_units` in table_id `file_0_view_0`, indexed by `customer_id`.
- $f_i$: `fixed_opening_cost` in table_id `file_1_view_0`, indexed by `facility_id`.
- $c_{ij}$: `transportation_cost_to_{j}` in table_id `file_2_view_0` (`transportation_costs.csv`), indexed by `facility_id` and customer $j$.
- $M_i$: $\sum_{j \in J} d_j$ (sum of all `demand_units` in table_id `file_0_view_0`).

No supplier capacity constraints are present beyond activation. All indices and parameters are defined by the full set of entities in the current CSVs.