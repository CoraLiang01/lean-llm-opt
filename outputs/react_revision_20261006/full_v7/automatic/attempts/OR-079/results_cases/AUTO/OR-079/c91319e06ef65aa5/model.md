##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Factory capacity (only if constructed):**
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameter Mapping

- $I$: set of factories, from column "Facility" in table_id file_0_view_0 (facility_costs.csv)
- $J$: set of distribution centers, from column "Destination" in table_id file_2_view_0 (demand_requirements.csv)
- $f_i$: fixed cost of factory $i$, from column "FixedCost" in table_id file_0_view_0
- $K_i$: capacity of factory $i$, from column "Capacity" in table_id file_0_view_0
- $d_j$: demand at distribution center $j$, from column "Demand" in table_id file_2_view_0
- $c_{ij}$: shipping cost per unit from factory $i$ to distribution center $j$, from table_id file_1_view_0 (shipping_costs.csv), with row "Origin" = $i$ and column $j$

##### Data Mapping

- Factories $I$: file_0_view_0, column "Facility"
- Distribution centers $J$: file_2_view_0, column "Destination"
- Fixed costs $f_i$: file_0_view_0, column "FixedCost"
- Factory capacities $K_i$: file_0_view_0, column "Capacity"
- Demands $d_j$: file_2_view_0, column "Demand"
- Shipping costs $c_{ij}$: file_1_view_0, row "Origin" = $i$, column $j$ (where $j$ matches "Destination" in file_2_view_0)

All index sets and parameters are defined exactly as in the source data. No values are enumerated here; see the Observation for all current data.