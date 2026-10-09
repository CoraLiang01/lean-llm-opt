##### Decision Variables

- $y_i \in \{0,1\}$: 1 if facility $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from facility $i \in I$ to distribution center $j \in J$ (continuous).

##### Parameters

- $f_i$: fixed cost of constructing facility $i$ (from column "FixedCost", table_id: file_0_view_0, key: "Facility").
- $c_{ij}$: shipping cost per unit from facility $i$ to distribution center $j$ (from table_id: file_1_view_0, row_id: "Origin", column_id: $j$).
- $d_j$: demand at distribution center $j$ (from column "Demand", table_id: file_2_view_0, key: "Destination").
- $K_i$: capacity of facility $i$ (from column "Capacity", table_id: file_0_view_0, key: "Facility").

- $I$: set of facilities (from "Facility", table_id: file_0_view_0).
- $J$: set of distribution centers (from "Destination", table_id: file_2_view_0).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Facility capacity (only if constructed):**
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All "Facility" values from table_id: file_0_view_0
- $J$: All "Destination" values from table_id: file_2_view_0
- $f_i$: "FixedCost" column, table_id: file_0_view_0, key: "Facility"
- $K_i$: "Capacity" column, table_id: file_0_view_0, key: "Facility"
- $c_{ij}$: table_id: file_1_view_0, row_id: "Origin" = $i$, column_id: $j$
- $d_j$: "Demand" column, table_id: file_2_view_0, key: "Destination"