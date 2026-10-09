##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i, \quad \forall i \in I, \forall j \in J$ (inactive suppliers cannot ship)
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from fixed_cost.csv, column "Unnamed: 2", table_id: file_1_view_0)
- $J$ = set of stores (from demand.csv, column "Customer", table_id: file_0_view_0)
- $d_j$ = demand for store $j$ (from demand.csv, column "demand", table_id: file_0_view_0)
- $f_i$ = fixed cost for supplier $i$ (from fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, row "Unnamed: 2" for $i$, column with header matching $j$, table_id: file_2_view_0)
- $D_j$ = demand for store $j$ (as above, used as a valid upper bound for $x_{ij}$ if $y_i=0$)

##### Data Mapping

- Suppliers $I$: file_1_view_0, column "Unnamed: 2"
- Stores $J$: file_0_view_0, column "Customer"
- Demand $d_j$: file_0_view_0, column "demand"
- Fixed cost $f_i$: file_1_view_0, column "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 2" (supplier $i$), column with header matching store $j$
- Activation constraint bound $D_j$: file_0_view_0, column "demand"

All index sets and parameters are defined directly from the current CSV data.