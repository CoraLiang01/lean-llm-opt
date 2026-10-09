##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from facility (plant) $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether facility $i$ is opened (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each customer’s demand must be fully met.)

2. **Facility capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]
   (A facility can only supply up to its capacity if opened; if not opened, it supplies nothing.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of facilities (plants), from column `facility_id` in `file_0_view_0` (cost.csv).
- $J$: set of customers, from column `customer_id` in `file_1_view_0` (demand.csv).
- $f_i$: fixed opening cost for facility $i$, from column `fixed_opening_cost` in `file_0_view_0`.
- $K_i$: capacity of facility $i$, from column `facility_capacity` in `file_0_view_0`.
- $c_{ij}$: per-unit transportation cost from facility $i$ to customer $j$, from column `transportation_cost_to_{j}` in `file_0_view_0` (where $j$ matches customer IDs).
- $d_j$: demand of customer $j$, from column `demand_units` in `file_1_view_0`.

##### Data Mapping

- Facilities $I$: `file_0_view_0`, column `facility_id`
- Customers $J$: `file_1_view_0`, column `customer_id`
- Fixed opening cost $f_i$: `file_0_view_0`, column `fixed_opening_cost`
- Facility capacity $K_i$: `file_0_view_0`, column `facility_capacity`
- Transportation cost $c_{ij}$: `file_0_view_0`, column `transportation_cost_to_{j}` (for each $j$ in $J$)
- Demand $d_j$: `file_1_view_0`, column `demand_units`