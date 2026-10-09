##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$ (if supplier $i$ is not open, it cannot supply any store)
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from fixed_cost.csv and transportation_costs.csv row labels)
- $J$ = set of stores (from demand.csv and transportation_costs.csv column labels)
- $d_j$ = demand for store $j$ (from demand.csv, column "demand", table_id: file_0_view_0)
- $f_i$ = fixed cost for supplier $i$ (from fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, table_id: file_2_view_0, row "Unnamed: 0" = $i$, column = $j$)
- $D_j$ = demand for store $j$ (used as a valid upper bound for $x_{ij}$ in constraint 2)

##### Data Mapping

- $I$: All unique supplier names from fixed_cost.csv ("Unnamed: 1", table_id: file_1_view_0) and transportation_costs.csv ("Unnamed: 0", table_id: file_2_view_0)
- $J$: All unique store names from demand.csv ("Customer", table_id: file_0_view_0) and transportation_costs.csv (column headers, table_id: file_2_view_0)
- $d_j$: demand.csv, column "demand", table_id: file_0_view_0, indexed by "Customer"
- $f_i$: fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0, indexed by "Unnamed: 1"
- $c_{ij}$: transportation_costs.csv, table_id: file_2_view_0, row "Unnamed: 0" = $i$, column = $j$
- $x_{ij}$, $y_i$: decision variables as defined above

All index sets and parameters are to be taken directly from the referenced columns and table_ids in the current CSV files.