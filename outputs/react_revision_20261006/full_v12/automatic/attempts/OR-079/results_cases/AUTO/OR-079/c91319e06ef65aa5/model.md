##### Decision Variables

$y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.

$x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Objective Function

$\min \sum_{i \in I} F_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Demand satisfaction at each distribution center:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$

2. Factory capacity (only if constructed):
   $$
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   $$

3. Variable domains:
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

##### Index Sets and Parameters

- $I$: set of factories (Facility, from facility_costs.csv, table_id: file_0_view_0)
- $J$: set of distribution centers (Destination, from demand_requirements.csv, table_id: file_2_view_0)
- $F_i$: fixed cost of constructing factory $i$ (FixedCost, file_0_view_0)
- $K_i$: capacity of factory $i$ (Capacity, file_0_view_0)
- $d_j$: demand at distribution center $j$ (Demand, file_2_view_0)
- $c_{ij}$: shipping cost per unit from factory $i$ to distribution center $j$ (shipping_costs.csv, table_id: file_1_view_0, row: Origin = $i$, column: $j$)

##### Data Mapping

- $I$: Facility (file_0_view_0, column "Facility")
- $F_i$: FixedCost (file_0_view_0, column "FixedCost")
- $K_i$: Capacity (file_0_view_0, column "Capacity")
- $J$: Destination (file_2_view_0, column "Destination")
- $d_j$: Demand (file_2_view_0, column "Demand")
- $c_{ij}$: shipping_costs.csv (file_1_view_0), row "Origin" = $i$, column $j$ (columns "B1"-"B8")

All index sets, parameters, and constraints are mapped directly to the provided CSV data as described above.