##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Customer demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Plant capacity (only if built):**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \cdot y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants, $I = \{\text{F1}, \ldots, \text{F15}\}$ (from cost.csv, column facility_id)
- $J$: set of customers, $J = \{\text{C1}, \ldots, \text{C15}\}$ (from demand.csv, column customer_id)

##### Parameters and Data Mapping

- $f_i$: fixed opening cost for plant $i$  
  - Data: cost.csv, column fixed_opening_cost, indexed by facility_id (table_id: file_0_view_0, column: fixed_opening_cost)
- $\text{cap}_i$: capacity of plant $i$  
  - Data: cost.csv, column facility_capacity, indexed by facility_id (table_id: file_0_view_0, column: facility_capacity)
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$  
  - Data: cost.csv, columns transportation_cost_to_C1 ... transportation_cost_to_C15, indexed by facility_id and customer (table_id: file_0_view_0, columns: transportation_cost_to_C1 ... transportation_cost_to_C15)
- $d_j$: demand of customer $j$  
  - Data: demand.csv, column demand_units, indexed by customer_id (table_id: file_1_view_0, column: demand_units)

##### Data Mapping

- Plants $I$: file_0_view_0, column facility_id
- Customers $J$: file_1_view_0, column customer_id
- $f_i$: file_0_view_0, column fixed_opening_cost
- $\text{cap}_i$: file_0_view_0, column facility_capacity
- $c_{ij}$: file_0_view_0, columns transportation_cost_to_C1 ... transportation_cost_to_C15
- $d_j$: file_1_view_0, column demand_units

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files.