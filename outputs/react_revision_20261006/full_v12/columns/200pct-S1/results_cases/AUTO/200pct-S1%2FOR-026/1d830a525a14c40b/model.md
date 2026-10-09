##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i$ to customer $j$ (continuous), for all $i \in I$, $j \in J$.
- $y_i \in \{0,1\}$: 1 if plant $i$ is built, 0 otherwise, for all $i \in I$.

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. **Demand satisfaction:**  
   $\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. **Plant capacity:**  
   $\sum_{j \in J} x_{ij} \leq \text{cap}_i \cdot y_i \quad \forall i \in I$

3. **Variable domains:**  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of plants, from `file_0_view_0.facility_id`
- $J$: set of customers, from `file_1_view_0.customer_id`
- $f_i$: fixed opening cost of plant $i$, from `file_0_view_0.fixed_opening_cost`
- $\text{cap}_i$: capacity of plant $i$, from `file_0_view_0.facility_capacity`
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from `file_0_view_0.transportation_cost_to_{j}`
- $d_j$: demand of customer $j$, from `file_1_view_0.demand_units`

##### Data Mapping

- Plants: $I$ = all `facility_id` in `file_0_view_0`
- Customers: $J$ = all `customer_id` in `file_1_view_0`
- $f_i$: `file_0_view_0.fixed_opening_cost` for plant $i$
- $\text{cap}_i$: `file_0_view_0.facility_capacity` for plant $i$
- $c_{ij}$: `file_0_view_0.transportation_cost_to_{j}` for plant $i$, customer $j$
- $d_j$: `file_1_view_0.demand_units` for customer $j$