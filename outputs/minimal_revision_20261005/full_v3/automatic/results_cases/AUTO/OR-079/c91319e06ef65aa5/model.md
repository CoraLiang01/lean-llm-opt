##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

##### Parameters

- $f_i$: fixed cost of constructing factory $i$ (from facility_costs.csv, column FixedCost, table_id: file_0_view_0).
- $s_i$: capacity of factory $i$ (from facility_costs.csv, column Capacity, table_id: file_0_view_0).
- $c_{ij}$: shipping cost per unit from factory $i$ to distribution center $j$ (from shipping_costs.csv, table_id: file_1_view_0, row_id_mapping: Origin, column_id_mapping: B1–B8).
- $d_j$: demand at distribution center $j$ (from demand_requirements.csv, column Demand, table_id: file_2_view_0).

Let $I$ be the set of factories (A1–A15, from facility_costs.csv), and $J$ the set of distribution centers (B1–B8, from demand_requirements.csv).

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
   \sum_{j \in J} x_{ij} \leq s_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $f_i$: file_0_view_0, column FixedCost, row_id_mapping: Facility
- $s_i$: file_0_view_0, column Capacity, row_id_mapping: Facility
- $c_{ij}$: file_1_view_0, row_id_mapping: Origin, column_id_mapping: B1–B8
- $d_j$: file_2_view_0, column Demand, row_id_mapping: Destination

- $I$: Facilities A1–A15 (file_0_view_0, column Facility)
- $J$: Distribution centers B1–B8 (file_2_view_0, column Destination)