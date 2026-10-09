##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $\sum_{j \in J} x_{ij} \leq M_i y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ is the set of suppliers (facility locations from fixed_cost.csv and transportation_costs.csv rows)
- $J$ is the set of stores (customers from demand.csv and columns from transportation_costs.csv)
- $c_{ij}$ is the transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, table_id: file_2_view_0, row $i$, column $j$)
- $f_i$ is the fixed cost for opening supplier $i$ (from fixed_cost.csv, table_id: file_1_view_0, column "fixed_costs")
- $d_j$ is the demand at store $j$ (from demand.csv, table_id: file_0_view_0, column "demand")
- $M_i$ is a sufficiently large upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$)

##### Data Mapping

- Suppliers $I$: facility names from fixed_cost.csv ("Unnamed: 1", table_id: file_1_view_0) and transportation_costs.csv ("Unnamed: 0", table_id: file_2_view_0, rows)
- Stores $J$: customer names from demand.csv ("Customer", table_id: file_0_view_0) and transportation_costs.csv (columns: "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT", table_id: file_2_view_0)
- $d_j$: demand.csv, table_id: file_0_view_0, column "demand", indexed by "Customer"
- $f_i$: fixed_cost.csv, table_id: file_1_view_0, column "fixed_costs", indexed by "Unnamed: 1"
- $c_{ij}$: transportation_costs.csv, table_id: file_2_view_0, row "Unnamed: 0" (supplier), column (store)
- $x_{ij}$, $y_i$: as defined above

All index sets and parameters are to be taken directly from the referenced columns and rows in the source files. No values are to be omitted or abbreviated.