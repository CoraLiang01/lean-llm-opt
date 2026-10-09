##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Parameters

- $f_i$: fixed cost of constructing factory $i$ (from facility_costs.csv, column "FixedCost", table_id: file_0_view_0).
- $c_{ij}$: shipping cost per unit from factory $i$ to distribution center $j$ (from shipping_costs.csv, row "Origin" = $i$, column $j$, table_id: file_1_view_0).
- $d_j$: demand at distribution center $j$ (from demand_requirements.csv, column "Demand", table_id: file_2_view_0).
- $u_i$: capacity of factory $i$ (from facility_costs.csv, column "Capacity", table_id: file_0_view_0).
- $I$: set of factories (from facility_costs.csv, column "Facility", table_id: file_0_view_0).
- $J$: set of distribution centers (from demand_requirements.csv, column "Destination", table_id: file_2_view_0).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Factory capacity:**
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All "Facility" values in facility_costs.csv (table_id: file_0_view_0, column "Facility")
- $J$: All "Destination" values in demand_requirements.csv (table_id: file_2_view_0, column "Destination")
- $f_i$: facility_costs.csv (table_id: file_0_view_0, column "FixedCost", indexed by "Facility")
- $u_i$: facility_costs.csv (table_id: file_0_view_0, column "Capacity", indexed by "Facility")
- $d_j$: demand_requirements.csv (table_id: file_2_view_0, column "Demand", indexed by "Destination")
- $c_{ij}$: shipping_costs.csv (table_id: file_1_view_0, row "Origin" = $i$, column $j$)

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files.