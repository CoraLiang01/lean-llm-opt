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
   x_{ij} \leq D_j y_i, \quad \forall i \in I,\, j \in J
   \]
   where $D_j$ is the demand of branch $j$ (from data), ensuring $x_{ij}=0$ if $y_i=0$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of suppliers (facility IDs from fixed_cost.csv and transportation_costs.csv)
- $J$: set of branches/customers (customer IDs from demand.csv and transportation_costs.csv)

##### Data Mapping

- $d_j$: demand of branch $j$  
  - Source: demand.csv, table_id: file_0_view_0, column: demand_units, row: customer_id = $j$
- $f_i$: fixed opening cost for supplier $i$  
  - Source: fixed_cost.csv, table_id: file_1_view_0, column: fixed_opening_cost, row: facility_id = $i$
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$  
  - Source: transportation_costs.csv, table_id: file_2_view_0, column: transportation_cost_to_$j$, row: facility_id = $i$
- $I$: all facility_id in fixed_cost.csv and transportation_costs.csv (table_id: file_1_view_0, file_2_view_0)
- $J$: all customer_id in demand.csv and all transportation_cost_to_* columns in transportation_costs.csv (table_id: file_0_view_0, file_2_view_0)

##### Notes

- All parameters are mapped directly to the CSV data as described above.
- No supplier capacity limits are specified, so only demand and activation logic are enforced.
- The model minimizes the sum of fixed opening and transportation costs.