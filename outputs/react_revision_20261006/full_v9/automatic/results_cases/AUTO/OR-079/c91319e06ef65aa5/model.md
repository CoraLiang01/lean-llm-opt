##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i \in I$ is constructed, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).

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
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of factories, from column "Facility" in facility_costs.csv (table_id: file_0_view_0)
- $J$: set of distribution centers, from column "Destination" in demand_requirements.csv (table_id: file_2_view_0)
- $f_i$: fixed cost of constructing factory $i$, from column "FixedCost" in facility_costs.csv (table_id: file_0_view_0)
- $K_i$: capacity of factory $i$, from column "Capacity" in facility_costs.csv (table_id: file_0_view_0)
- $d_j$: demand at distribution center $j$, from column "Demand" in demand_requirements.csv (table_id: file_2_view_0)
- $c_{ij}$: per-unit shipping cost from factory $i$ to distribution center $j$, from shipping_costs.csv (table_id: file_1_view_0), with rows indexed by "Origin" (factories) and columns by distribution center IDs.

##### Data Mapping

- Factories $I$: facility_costs.csv, table_id: file_0_view_0, column "Facility"
- Distribution centers $J$: demand_requirements.csv, table_id: file_2_view_0, column "Destination"
- Fixed costs $f_i$: facility_costs.csv, table_id: file_0_view_0, column "FixedCost"
- Factory capacities $K_i$: facility_costs.csv, table_id: file_0_view_0, column "Capacity"
- Demands $d_j$: demand_requirements.csv, table_id: file_2_view_0, column "Demand"
- Shipping costs $c_{ij}$: shipping_costs.csv, table_id: file_1_view_0, rows "Origin" (factories), columns distribution center IDs

All index sets, parameters, and constraints are defined directly from the current CSV data. No values are enumerated here; all mappings are symbolic and reference the exact source columns and table_ids.