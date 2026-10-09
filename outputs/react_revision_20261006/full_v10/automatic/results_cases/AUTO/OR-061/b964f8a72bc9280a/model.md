##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i$

##### Constraints

1. Branch demand: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq U_{ij} y_i,\quad \forall i \in I,\, j \in J$ (where $U_{ij}$ is a sufficiently large upper bound, e.g., $U_{ij} = \sum_{j \in J} d_j$)
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

##### Index Sets

$I = \{$S1, S2, S3, S4, S5$\}$ (suppliers, from fixed_cost.csv and transportation_costs.csv, table_id: file_1_view_0 and file_2_view_0, column: Unnamed: 0)  
$J = \{$C1, C2, C3, C4, C5$\}$ (branches, from demand.csv and transportation_costs.csv, table_id: file_0_view_0 and file_2_view_0, column: customer)

##### Parameters and Data Mapping

- $d_j$: demand of branch $j$ (from demand.csv, table_id: file_0_view_0, column: demand, indexed by customer)
- $f_i$: fixed cost for supplier $i$ (from fixed_cost.csv, table_id: file_1_view_0, column: fixed_costs, indexed by Unnamed: 0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$ (from transportation_costs.csv, table_id: file_2_view_0, row: Unnamed: 0, columns: C1–C5)
- $U_{ij}$: upper bound for $x_{ij}$, set to $\sum_{j \in J} d_j$ (sum of all demands, from demand.csv, table_id: file_0_view_0, column: demand)

##### Data Mapping

- $I$: file_1_view_0, column: Unnamed: 0; file_2_view_0, row: Unnamed: 0
- $J$: file_0_view_0, column: customer; file_2_view_0, columns: C1–C5
- $d_j$: file_0_view_0, columns: customer, demand
- $f_i$: file_1_view_0, columns: Unnamed: 0, fixed_costs
- $c_{ij}$: file_2_view_0, row: Unnamed: 0, columns: C1–C5

All index sets, parameters, and coefficients are mapped directly from the CSV files as described above.