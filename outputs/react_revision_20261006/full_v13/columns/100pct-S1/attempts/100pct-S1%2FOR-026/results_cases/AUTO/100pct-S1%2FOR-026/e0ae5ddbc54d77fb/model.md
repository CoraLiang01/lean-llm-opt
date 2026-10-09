##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$ (from cost.csv, columns transportation_cost_to_C1, ..., transportation_cost_to_C15, table_id: file_0_view_0)
- $f_i$: fixed opening cost for plant $i$ (from cost.csv, column fixed_opening_cost, table_id: file_0_view_0)

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   where $d_j$ is the demand of customer $j$ (from demand.csv, column demand_units, table_id: file_1_view_0)

2. **Plant capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]
   where $K_i$ is the capacity of plant $i$ (from cost.csv, column facility_capacity, table_id: file_0_view_0)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Index Sets

- $I$: set of plants (facility_id in cost.csv, table_id: file_0_view_0)
- $J$: set of customers (customer_id in demand.csv, table_id: file_1_view_0)

##### Data Mapping

- $f_i$: cost.csv, column fixed_opening_cost, indexed by facility_id, table_id: file_0_view_0
- $K_i$: cost.csv, column facility_capacity, indexed by facility_id, table_id: file_0_view_0
- $c_{ij}$: cost.csv, columns transportation_cost_to_C1, ..., transportation_cost_to_C15, indexed by facility_id and customer $j$, table_id: file_0_view_0
- $d_j$: demand.csv, column demand_units, indexed by customer_id, table_id: file_1_view_0
- $I$: all facility_id in cost.csv, table_id: file_0_view_0
- $J$: all customer_id in demand.csv, table_id: file_1_view_0