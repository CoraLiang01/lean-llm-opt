##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i$

##### Constraints

1. Demand satisfaction: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Warehouse activation: $\sum_{j\in J} x_{ij} \leq M y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where $M = \sum_{j\in J} d_j$ is a valid upper bound on total shipments from any warehouse, since there are no explicit warehouse capacity limits.

##### Index Sets and Data Mapping

- $I$: set of warehouses, from column "Unnamed: 0" in table_id file_1_view_0 (fixed_cost.csv)
- $J$: set of musicians/bands, from column "customer" in table_id file_0_view_0 (demand.csv)
- $d_j$: demand of musician/band $j$, from column "demand" in table_id file_0_view_0 (demand.csv)
- $f_i$: fixed cost for warehouse $i$, from column "fixed_costs" in table_id file_1_view_0 (fixed_cost.csv)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to musician/band $j$, from table_id file_2_view_0 (transportation_costs.csv), with rows indexed by "Unnamed: 0" (warehouses) and columns by "customer" (musicians/bands)
- $M = \sum_{j\in J} d_j$, computed from demand.csv

All parameters are mapped directly from the source data as described above.