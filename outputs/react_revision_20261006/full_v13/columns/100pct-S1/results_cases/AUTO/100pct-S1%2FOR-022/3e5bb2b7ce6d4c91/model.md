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
   where $U_{ij}$ is a sufficiently large upper bound (e.g., $U_{ij} = d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers (facility_id from file_1_view_0 and file_2_view_0)
- $J$: set of branches/customers (customer_id from file_0_view_0 and columns transportation_cost_to_C* in file_2_view_0)
- $d_j$: demand of branch $j$ (demand_units from file_0_view_0)
- $f_i$: fixed opening cost for supplier $i$ (fixed_opening_cost from file_1_view_0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$ (transportation_cost_to_C* columns from file_2_view_0)
- $U_{ij}$: upper bound for $x_{ij}$, set to $d_j$ for each $j$.

##### Data Mapping

- $I$: All facility_id in file_1_view_0 and file_2_view_0
- $J$: All customer_id in file_0_view_0 and all columns with suffix transportation_cost_to_C* in file_2_view_0
- $d_j$: demand_units column in file_0_view_0, indexed by customer_id
- $f_i$: fixed_opening_cost column in file_1_view_0, indexed by facility_id
- $c_{ij}$: transportation_cost_to_C* columns in file_2_view_0, indexed by facility_id (rows) and customer_id (columns)
- $U_{ij}$: $d_j$ for each $j$ (from file_0_view_0)

All index sets and parameters are defined by the full set of entities in the current CSV files.