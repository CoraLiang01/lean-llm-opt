##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i$

##### Constraints

1. Supermarket demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $\sum_{j\in J} x_{ij} \leq M_i y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from fixed_cost.csv, table_id: file_1_view_0, column: Unnamed: 0)
- $J$ = set of supermarkets (from demand.csv, table_id: file_0_view_0, column: customer)
- $d_j$ = demand of supermarket $j$ (from demand.csv, table_id: file_0_view_0, column: demand)
- $f_i$ = fixed cost for supplier $i$ (from fixed_cost.csv, table_id: file_1_view_0, column: fixed_costs)
- $c_{ij}$ = per-unit transportation cost from supplier $i$ to supermarket $j$ (from transportation_costs.csv, table_id: file_2_view_0, row: Unnamed: 1, columns: $J$)
- $M_i$ = $\sum_{j\in J} d_j$ (total demand; valid upper bound since no supplier capacity is specified)

##### Data Mapping

- $I$: supplier IDs from fixed_cost.csv (table_id: file_1_view_0, column: Unnamed: 0)
- $J$: supermarket IDs from demand.csv (table_id: file_0_view_0, column: customer)
- $d_j$: demand.csv (table_id: file_0_view_0, columns: customer, demand)
- $f_i$: fixed_cost.csv (table_id: file_1_view_0, columns: Unnamed: 0, fixed_costs)
- $c_{ij}$: transportation_costs.csv (table_id: file_2_view_0, row: Unnamed: 1, columns: $J$)
- $M_i$: $\sum_{j\in J} d_j$ (computed from demand.csv, table_id: file_0_view_0, column: demand)