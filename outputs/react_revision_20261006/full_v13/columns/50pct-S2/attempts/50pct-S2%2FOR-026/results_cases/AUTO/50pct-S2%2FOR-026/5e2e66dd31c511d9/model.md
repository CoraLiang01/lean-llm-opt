##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Plant capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \, y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants, $I = \{\text{F1}, \ldots, \text{F15}\}$ (from cost.csv, column facility_id)
- $J$: set of customers, $J = \{\text{C1}, \ldots, \text{C15}\}$ (from demand.csv, column customer_id)

##### Parameters and Data Mapping

- $f_i$: fixed opening cost of plant $i$  
  (cost.csv, table_id: file_0_view_0, column: fixed_opening_cost, key: facility_id)
- $\text{cap}_i$: capacity of plant $i$  
  (cost.csv, table_id: file_0_view_0, column: facility_capacity, key: facility_id)
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$  
  (cost.csv, table_id: file_0_view_0, columns: transportation_cost_to_C1 ... transportation_cost_to_C15, key: facility_id)
- $d_j$: demand of customer $j$  
  (demand.csv, table_id: file_1_view_0, column: demand_units, key: customer_id)

##### Data Mapping

- Plants $I$ and their parameters:  
  - Table: cost.csv (table_id: file_0_view_0)  
  - Columns: facility_id, fixed_opening_cost, facility_capacity, transportation_cost_to_C1 ... transportation_cost_to_C15

- Customers $J$ and their demands:  
  - Table: demand.csv (table_id: file_1_view_0)  
  - Columns: customer_id, demand_units

- $c_{ij}$:  
  - Table: cost.csv (table_id: file_0_view_0)  
  - Row: facility_id = $i$  
  - Column: transportation_cost_to_$j$ (e.g., transportation_cost_to_C1 for $j$ = C1)

- $f_i$, $\text{cap}_i$:  
  - Table: cost.csv (table_id: file_0_view_0)  
  - Row: facility_id = $i$  
  - Columns: fixed_opening_cost, facility_capacity

- $d_j$:  
  - Table: demand.csv (table_id: file_1_view_0)  
  - Row: customer_id = $j$  
  - Column: demand_units

##### Summary

This is a capacitated facility location problem with fixed opening costs, plant capacities, and customer demands, using the exact symbolic parameters and data mapping as above.