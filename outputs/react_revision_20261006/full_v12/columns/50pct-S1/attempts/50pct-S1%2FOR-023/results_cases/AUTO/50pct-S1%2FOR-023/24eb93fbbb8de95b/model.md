##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (binary)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $\sum_{j \in J} x_{ij} \leq M_i y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from fixed_cost.csv and transportation_costs.csv, e.g., {"MOUNT AYR", "WAUKEE", "WAVERLY", "PELLA", "DES MOINES"})
- $J$ = set of stores (from demand.csv and transportation_costs.csv, e.g., {"CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"})
- $d_j$ = demand for store $j$ (from demand.csv, table_id: file_0_view_0, column: demand, key: Customer)
- $f_i$ = fixed cost for supplier $i$ (from fixed_cost.csv, table_id: file_1_view_0, column: fixed_costs, key: Unnamed: 1)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, table_id: file_2_view_0, row: Unnamed: 0, column: store name)
- $M_i$ = $\sum_{j \in J} d_j$ (a valid upper bound for each supplier, since there are no explicit supplier capacity limits)

##### Data Mapping

- $I$: supplier names from fixed_cost.csv (file_1_view_0, Unnamed: 1) and transportation_costs.csv (file_2_view_0, Unnamed: 0)
- $J$: store names from demand.csv (file_0_view_0, Customer) and transportation_costs.csv (file_2_view_0, columns CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT)
- $d_j$: file_0_view_0, column demand, key Customer
- $f_i$: file_1_view_0, column fixed_costs, key Unnamed: 1
- $c_{ij}$: file_2_view_0, row Unnamed: 0 (supplier), column (store)
- $M_i$: $\sum_{j \in J} d_j$ (computed from file_0_view_0, column demand)

All index sets and parameters are defined by the union of the relevant identifiers in the current CSV files.