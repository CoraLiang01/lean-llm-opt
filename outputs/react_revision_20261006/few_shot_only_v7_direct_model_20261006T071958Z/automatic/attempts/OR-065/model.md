##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Demand satisfaction: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Warehouse activation: $\sum_{j \in J} x_{ij} \leq M_i y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ = set of warehouses (from fixed_cost.csv: column "Unnamed: 0", table_id "file_1_view_0")
- $J$ = set of musicians/bands (from demand.csv: column "customer", table_id "file_0_view_0")
- $d_j$ = demand of musician/band $j$ (from demand.csv: column "demand", table_id "file_0_view_0")
- $f_i$ = fixed cost for warehouse $i$ (from fixed_cost.csv: column "fixed_costs", table_id "file_1_view_0")
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to musician/band $j$ (from transportation_costs.csv: row "Unnamed: 0" = $i$, column $j$, table_id "file_2_view_0")
- $M_i$ = $\sum_{j \in J} d_j$ (since no explicit warehouse capacity is given, $M_i$ is set to total demand as a valid upper bound)

##### Data Mapping

- Warehouses $I$: table_id "file_1_view_0", column "Unnamed: 0"
- Musicians/Bands $J$: table_id "file_0_view_0", column "customer"
- Demand $d_j$: table_id "file_0_view_0", columns "customer", "demand"
- Fixed cost $f_i$: table_id "file_1_view_0", columns "Unnamed: 0", "fixed_costs"
- Transportation cost $c_{ij}$: table_id "file_2_view_0", row "Unnamed: 0" = $i$, columns $j$
- $M_i$: $\sum_{j \in J} d_j$ (computed from demand.csv, table_id "file_0_view_0", column "demand")