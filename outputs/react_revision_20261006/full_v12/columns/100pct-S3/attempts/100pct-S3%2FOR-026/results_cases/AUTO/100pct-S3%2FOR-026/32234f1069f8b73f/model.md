##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$

2. **Plant capacity:**  
   $\sum_{j \in J} x_{ij} \leq K_i y_i,\quad \forall i \in I$

3. **Variable domains:**  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of plants, from `file_0_view_0`, column `facility_id`
- $J$: set of customers, from `file_1_view_0`, column `customer_id`
- $f_i$: fixed opening cost for plant $i$, from `file_0_view_0`, column `fixed_opening_cost`
- $K_i$: capacity of plant $i$, from `file_0_view_0`, column `facility_capacity`
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from `file_0_view_0`, column `transportation_cost_to_{j}` (where $j$ matches customer_id in $J$)
- $d_j$: demand of customer $j$, from `file_1_view_0`, column `demand_units`