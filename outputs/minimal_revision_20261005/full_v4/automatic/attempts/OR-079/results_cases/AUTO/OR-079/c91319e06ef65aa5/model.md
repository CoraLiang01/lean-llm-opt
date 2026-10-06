##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Parameters

- $f_i$: fixed cost of constructing factory $i$ (from column "FixedCost", table_id: file_0_view_0).
- $c_{ij}$: shipping cost per unit from factory $i$ to distribution center $j$ (from table_id: file_1_view_0, row "Origin" = $i$, column $j$).
- $d_j$: demand at distribution center $j$ (from column "Demand", table_id: file_2_view_0).
- $u_i$: capacity of factory $i$ (from column "Capacity", table_id: file_0_view_0).

- $I$: set of factories (from column "Facility", table_id: file_0_view_0).
- $J$: set of distribution centers (from column "Destination", table_id: file_2_view_0).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Factory capacity (only if constructed):**
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: All values in column "Facility", table_id: file_0_view_0.
- $J$: All values in column "Destination", table_id: file_2_view_0.
- $f_i$: "FixedCost" for facility $i$, table_id: file_0_view_0.
- $u_i$: "Capacity" for facility $i$, table_id: file_0_view_0.
- $d_j$: "Demand" for destination $j$, table_id: file_2_view_0.
- $c_{ij}$: Value at row "Origin" = $i$, column $j$ in table_id: file_1_view_0, with $j$ matching "Destination" in file_2_view_0.

No values are omitted; all index sets and parameters are defined directly from the CSV data.