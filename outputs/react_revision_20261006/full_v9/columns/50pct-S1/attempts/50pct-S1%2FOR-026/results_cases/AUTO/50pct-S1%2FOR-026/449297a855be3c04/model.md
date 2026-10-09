##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$ (from cost.csv, columns transportation_cost_to_C1 ... transportation_cost_to_C15, table_id: file_0_view_0)
- $f_i$: fixed opening cost for plant $i$ (from cost.csv, column fixed_opening_cost, table_id: file_0_view_0)

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   where $d_j$ is the demand of customer $j$ (from demand.csv, column demand_units, table_id: file_1_view_0)

2. **Plant capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq s_i y_i, \quad \forall i \in I
   \]
   where $s_i$ is the capacity of plant $i$ (from cost.csv, column facility_capacity, table_id: file_0_view_0)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants, $I = \{$facility_id from cost.csv, table_id: file_0_view_0$\}$
- $J$: set of customers, $J = \{$customer_id from demand.csv, table_id: file_1_view_0$\}$

##### Data Mapping

- $f_i$: file_0_view_0, column fixed_opening_cost, indexed by facility_id
- $s_i$: file_0_view_0, column facility_capacity, indexed by facility_id
- $c_{ij}$: file_0_view_0, columns transportation_cost_to_C1 ... transportation_cost_to_C15, indexed by facility_id and customer $j$
- $d_j$: file_1_view_0, column demand_units, indexed by customer_id

All index sets, parameters, and constraints are defined directly from the current CSV data. No values are enumerated here; all mappings are symbolic and refer to the exact source columns and table_ids.