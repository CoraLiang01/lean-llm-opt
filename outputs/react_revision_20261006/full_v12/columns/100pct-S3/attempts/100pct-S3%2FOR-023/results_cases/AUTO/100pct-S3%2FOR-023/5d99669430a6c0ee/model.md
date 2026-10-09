##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of suppliers (from column "Unnamed: 2" in fixed_cost.csv and transportation_costs.csv, table_id file_1_view_0 and file_2_view_0)
- $J$: set of stores (from column "Customer" in demand.csv and columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"] in transportation_costs.csv, table_id file_0_view_0 and file_2_view_0)
- $d_j$: demand for store $j$ (from column "demand" in demand.csv, table_id file_0_view_0)
- $f_i$: fixed cost for supplier $i$ (from column "fixed_costs" in fixed_cost.csv, table_id file_1_view_0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from table_id file_2_view_0, row "Unnamed: 2" = $i$, column $j$)

##### Data Mapping

- $I$: file_1_view_0["Unnamed: 2"], file_2_view_0["Unnamed: 2"]
- $J$: file_0_view_0["Customer"], file_2_view_0 columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"]
- $d_j$: file_0_view_0["demand"]
- $f_i$: file_1_view_0["fixed_costs"]
- $c_{ij}$: file_2_view_0, row "Unnamed: 2" = $i$, column $j$ (among ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"])

No supplier capacity or activation-conditioned bounds are specified; all $x_{ij}$ are nonnegative and unconstrained except by demand satisfaction.