##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Plant capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: set of plants, from column `facility_id` in table_id `file_0_view_0` (`cost.csv`)
- $J$: set of customers, from column `customer_id` in table_id `file_1_view_0` (`demand.csv`)
- $f_i$: fixed opening cost for plant $i$, from column `fixed_opening_cost` in table_id `file_0_view_0`
- $u_i$: capacity of plant $i$, from column `facility_capacity` in table_id `file_0_view_0`
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from column `transportation_cost_to_{j}` in table_id `file_0_view_0` (where $j$ matches customer_id in $J$)
- $d_j$: demand of customer $j$, from column `demand_units` in table_id `file_1_view_0`

##### Data Mapping

- Plants: `facility_id` in `file_0_view_0`
- Customers: `customer_id` in `file_1_view_0`
- Fixed opening cost $f_i$: `fixed_opening_cost` in `file_0_view_0`
- Plant capacity $u_i$: `facility_capacity` in `file_0_view_0`
- Transportation cost $c_{ij}$: `transportation_cost_to_{j}` in `file_0_view_0`, with $j$ as in `customer_id` of `file_1_view_0`
- Demand $d_j$: `demand_units` in `file_1_view_0`