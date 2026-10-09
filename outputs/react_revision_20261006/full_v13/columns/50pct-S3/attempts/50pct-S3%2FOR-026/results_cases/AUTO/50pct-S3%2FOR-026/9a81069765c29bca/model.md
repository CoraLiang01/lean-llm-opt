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

1. **Demand satisfaction** (each customer must receive exactly its demand):

   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

   where $d_j$ is the demand of customer $j$ (from demand.csv, column demand_units, table_id: file_1_view_0).

2. **Plant capacity** (no plant can supply more than its capacity):

   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]

   where $u_i$ is the capacity of plant $i$ (from cost.csv, column facility_capacity, table_id: file_0_view_0).

3. **Variable domains**:

   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Index Sets

- $I$: set of plants (facility_id from cost.csv, table_id: file_0_view_0)
- $J$: set of customers (customer_id from demand.csv, table_id: file_1_view_0)

##### Data Mapping

- $I$: All facility_id in cost.csv (table_id: file_0_view_0)
- $J$: All customer_id in demand.csv (table_id: file_1_view_0)
- $f_i$: fixed_opening_cost, cost.csv, table_id: file_0_view_0, indexed by facility_id
- $u_i$: facility_capacity, cost.csv, table_id: file_0_view_0, indexed by facility_id
- $c_{ij}$: transportation_cost_to_Cj, cost.csv, table_id: file_0_view_0, indexed by facility_id and customer_id (C1–C15)
- $d_j$: demand_units, demand.csv, table_id: file_1_view_0, indexed by customer_id

All parameters and index sets are defined by the full set of records in the respective CSV files.