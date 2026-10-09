##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

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
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \, y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants, from `file_0_view_0`, column `facility_id`
- $J$: set of customers, from `file_1_view_0`, column `customer_id`

##### Parameters and Data Mapping

- $f_i$: fixed opening cost of plant $i$, from `file_0_view_0`, column `fixed_opening_cost`
- $\text{cap}_i$: capacity of plant $i$, from `file_0_view_0`, column `facility_capacity`
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from `file_0_view_0`, column `transportation_cost_to_{j}` (where $j$ matches customer_id in $J$)
- $d_j$: demand of customer $j$, from `file_1_view_0`, column `demand_units`

##### Data Mapping

- Plants $I$: `file_0_view_0`, column `facility_id`
- Customers $J$: `file_1_view_0`, column `customer_id`
- Fixed opening cost $f_i$: `file_0_view_0`, column `fixed_opening_cost`
- Plant capacity $\text{cap}_i$: `file_0_view_0`, column `facility_capacity`
- Transportation cost $c_{ij}$: `file_0_view_0`, column `transportation_cost_to_{j}` (with $j$ as in $J$)
- Demand $d_j$: `file_1_view_0`, column `demand_units`