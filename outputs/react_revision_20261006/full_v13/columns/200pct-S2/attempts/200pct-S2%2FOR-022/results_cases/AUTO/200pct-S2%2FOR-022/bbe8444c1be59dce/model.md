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
   where $M_i = \sum_{j \in J} d_j$ is a valid upper bound for each supplier (since there are no explicit supplier capacity limits).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
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
  — Source: `transportation_costs.csv`, table_id: file_2_view_0, row: facility_id, columns: transportation_cost_to_C1, ..., transportation_cost_to_C5 (column_id_mapping: {"transportation_cost_to_C1": "C1", ..., "transportation_cost_to_C5": "C5"})
- $M_i$: big-M upper bound for each supplier $i$ (set to $\sum_{j \in J} d_j$)

##### Data Mapping

- $I = \{$facility_id$\}$ from `fixed_cost.csv` (file_1_view_0) and `transportation_costs.csv` (file_2_view_0)
- $J = \{$customer_id$\}$ from `demand.csv` (file_0_view_0) and columns of `transportation_costs.csv` (file_2_view_0)
- $d_j$: file_0_view_0, columns: customer_id, demand_units
- $f_i$: file_1_view_0, columns: facility_id, fixed_opening_cost
- $c_{ij}$: file_2_view_0, row: facility_id, columns: transportation_cost_to_C1, ..., transportation_cost_to_C5 (column_id_mapping: {"transportation_cost_to_C1": "C1", ..., "transportation_cost_to_C5": "C5"})
- $M_i = \sum_{j \in J} d_j$ (computed from file_0_view_0, demand_units)

No supplier capacity limits are present; $M_i$ is a valid upper bound for all $i$.

##### Matrix Structure

- Rows: suppliers (facility_id)
- Columns: branches/customers (customer_id, mapped via column_id_mapping)

All parameters and index sets are defined directly from the current CSV data.