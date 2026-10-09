##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Musician/band demand: $\sum_{i\in I}x_{ij}=d_j,\quad \forall j\in J$.
2. Warehouse activation: $\sum_{j\in J}x_{ij}\leq M y_i,\quad \forall i\in I$.
3. Domains: $x_{ij}\geq0$ continuous; $y_i\in\{0,1\}$.

Where:
- $I$ = set of warehouses (from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0)
- $J$ = set of musicians/bands (from column "customer" in table_id file_0_view_0 and columns "C1", "C2", "C3" in file_2_view_0)
- $d_j$ = demand of musician/band $j$ (from column "demand" in table_id file_0_view_0)
- $f_i$ = fixed cost of warehouse $i$ (from column "fixed_costs" in table_id file_1_view_0)
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to musician/band $j$ (from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$)
- $M = \sum_{j\in J} d_j$ (total demand, used as a valid upper bound for each warehouse's possible shipments)

##### Data Mapping

- $I$: All values in "Unnamed: 0" column of file_1_view_0 and file_2_view_0 (warehouses)
- $J$: All values in "customer" column of file_0_view_0 and columns "C1", "C2", "C3" of file_2_view_0 (musicians/bands)
- $d_j$: "demand" column in file_0_view_0, indexed by "customer"
- $f_i$: "fixed_costs" column in file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $M$: $\sum_{j\in J} d_j$ (sum over "demand" in file_0_view_0)

All indices, parameters, and constraints are mapped directly to the provided CSV data.