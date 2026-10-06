##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I$: Set of suppliers, from `file_1_view_0.facility_id` and `file_2_view_0.facility_id`.
- $J$: Set of branches, from `file_0_view_0.customer_id` and `file_2_view_0` column suffixes.
- $d_j$: Demand at branch $j$, from `file_0_view_0.demand_units`.
- $f_i$: Fixed opening cost for supplier $i$, from `file_1_view_0.fixed_opening_cost`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$, from `file_2_view_0.transportation_cost_to_{j}`.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch Demand Satisfaction**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier Activation**  
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand), ensuring inactive suppliers do not ship goods.

3. **Variable Domains**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: All `facility_id` in `file_1_view_0` and `file_2_view_0`.
- $J$: All `customer_id` in `file_0_view_0` and all columns with suffix `transportation_cost_to_{j}` in `file_2_view_0`.
- $d_j$: `file_0_view_0.demand_units` for each `customer_id`.
- $f_i$: `file_1_view_0.fixed_opening_cost` for each `facility_id`.
- $c_{ij}$: `file_2_view_0.transportation_cost_to_{j}` for each `facility_id` and `customer_id`.
- $M$: $\sum_{j \in J} d_j$ using all `file_0_view_0.demand_units`.

All index sets and parameters are defined directly from the CSV data as described above.