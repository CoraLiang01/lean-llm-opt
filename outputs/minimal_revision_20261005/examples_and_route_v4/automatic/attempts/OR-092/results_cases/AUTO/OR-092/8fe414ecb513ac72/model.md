##### Decision Variables

Let $x_{ij} \geq 0$ be the number of units shipped from source $i \in I$ to destination $j \in J$ (continuous, $0 \leq x_{ij} \leq 10 y_{ij}$).

Let $y_{ij} \in \mathbb{Z}_+$ be the number of trucks dispatched from source $i$ to destination $j$ (integer, $y_{ij} \geq 0$).

##### Sets

- $I$ = set of sources = $\{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10$\}$ (from expanded_sources.csv)
- $J$ = set of destinations = $\{$D1, D2, ..., D20$\}$ (from expanded_destinations.csv)

##### Parameters

- $c_{ij}$ = unit transportation cost from source $i$ to destination $j$ (from expanded_cost_matrix.csv, table_id: file_0_view_0, columns: D1-D20, row: source_id)
- $s_i$ = supply capacity at source $i$ (from expanded_sources.csv, table_id: file_2_view_0, column: supply_units, row: source_id)
- $d_j$ = demand at destination $j$ (from expanded_destinations.csv, table_id: file_1_view_0, column: demand_units, row: destination_id)
- Truck capacity = 10 units (from query)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each destination's demand must be met):
   $$
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   $$
2. **Supply capacity** (each source's shipments cannot exceed its supply):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Truck loading and integer dispatch** (each route's shipment must fit in the dispatched trucks, each truck can carry up to 10 units, and the number of trucks is integer):
   $$
   x_{ij} \leq 10 y_{ij} \qquad \forall i \in I,\, j \in J
   $$
   $$
   y_{ij} \in \mathbb{Z}_+, \quad x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (sources): All source_id in expanded_sources.csv (table_id: file_2_view_0, column: source_id)
- $J$ (destinations): All destination_id in expanded_destinations.csv (table_id: file_1_view_0, column: destination_id)
- $c_{ij}$: expanded_cost_matrix.csv (table_id: file_0_view_0, row: source_id $i$, column: $j$)
- $s_i$: expanded_sources.csv (table_id: file_2_view_0, row: source_id $i$, column: supply_units)
- $d_j$: expanded_destinations.csv (table_id: file_1_view_0, row: destination_id $j$, column: demand_units)

##### Summary

- Decision variables: $x_{ij} \geq 0$ (continuous), $y_{ij} \in \mathbb{Z}_+$
- Objective: $\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$
- Constraints:
  - $\sum_{i \in I} x_{ij} = d_j$ $\forall j \in J$
  - $\sum_{j \in J} x_{ij} \leq s_i$ $\forall i \in I$
  - $x_{ij} \leq 10 y_{ij}$ $\forall i \in I,\, j \in J$
  - $y_{ij} \in \mathbb{Z}_+$, $x_{ij} \geq 0$ $\forall i \in I,\, j \in J$
- All parameters and sets are mapped directly to the provided CSV files and columns as specified above.