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
   where $M_i$ is a sufficiently large upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}, \quad \forall i \in I,\, j \in J
   \]

##### Index Sets

- $I$: set of suppliers (facility IDs from `fixed_cost.csv` and `transportation_costs.csv`)
- $J$: set of branches/customers (customer IDs from `demand.csv` and columns of `transportation_costs.csv`)

##### Parameters and Data Mapping

- $d_j$: demand of branch $j$  
  — Source: `demand.csv`, table_id: file_0_view_0, columns: customer_id, demand_units
- $f_i$: fixed opening cost for supplier $i$  
  — Source: `fixed_cost.csv`, table_id: file_1_view_0, columns: facility_id, fixed_opening_cost
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$  
  — Source: `transportation_costs.csv`, table_id: file_2_view_0, row: facility_id, columns: transportation_cost_to_Ck (mapping: Ck = customer_id in demand.csv)
- $M_i$: big-M for each supplier $i$ (set as $M_i = \sum_{j \in J} d_j$ unless a tighter bound is available from data)

##### Data Mapping

- $I$: All facility_id in file_1_view_0 and file_2_view_0
- $J$: All customer_id in file_0_view_0 and all transportation_cost_to_Ck columns in file_2_view_0
- $d_j$: file_0_view_0, columns: customer_id, demand_units
- $f_i$: file_1_view_0, columns: facility_id, fixed_opening_cost
- $c_{ij}$: file_2_view_0, row: facility_id, columns: transportation_cost_to_Ck (Ck = customer_id)
- $M_i$: $M_i = \sum_{j \in J} d_j$ (sum over demand_units in file_0_view_0)

No additional capacity or supply constraints are imposed unless present in the data. All indices and parameters are mapped directly from the CSV sources as described.