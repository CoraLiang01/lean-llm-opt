##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Branch demand: $\sum_{i\in I}x_{ij}=d_j,\quad \forall j\in J$.
2. Supplier activation: $\sum_{j\in J}x_{ij}\leq M y_i,\quad \forall i\in I$.
3. Domains: $x_{ij}\geq0$ continuous; $y_i\in\{0,1\}$.

Where:
- $I$ = set of suppliers (from fixed_cost.csv, column "Unnamed: 0", table_id: file_1_view_0)
- $J$ = set of branches (from demand.csv, column "customer", table_id: file_0_view_0)
- $d_j$ = demand of branch $j$ (from demand.csv, column "demand", table_id: file_0_view_0)
- $f_i$ = fixed cost for supplier $i$ (from fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to branch $j$ (from transportation_costs.csv, columns "C1"..."C5", table_id: file_2_view_0, rows indexed by "Unnamed: 0")
- $M = \sum_{j\in J} d_j$ (total demand, computed from demand.csv, column "demand", table_id: file_0_view_0)

##### Data Mapping

- Suppliers $I$: file_1_view_0, column "Unnamed: 0"
- Branches $J$: file_0_view_0, column "customer"
- Demand $d_j$: file_0_view_0, columns "customer", "demand"
- Fixed cost $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, rows "Unnamed: 0" (supplier), columns "C1"..."C5" (branch)
- $M$: sum of file_0_view_0, column "demand"