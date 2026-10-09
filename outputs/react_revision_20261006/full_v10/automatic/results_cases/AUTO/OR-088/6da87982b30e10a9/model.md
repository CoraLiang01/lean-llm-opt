##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$

2. Plant capacity:  
   $\sum_{j \in J} x_{ij} \leq \text{cap}_i y_i,\quad \forall i \in I$

3. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of plants, from column "plant" in table_id file_0_view_0 (cost.csv)
- $J$: set of customers, from column "customer" in table_id file_1_view_0 (demand.csv)
- $f_i$: fixed_cost for plant $i$, from column "fixed_cost" in table_id file_0_view_0
- $\text{cap}_i$: capacity for plant $i$, from column "capacity" in table_id file_0_view_0
- $c_{ij}$: per-unit transport cost from plant $i$ to customer $j$, from columns "C1"–"C15" in table_id file_0_view_0, with $i$ from "plant" and $j$ from column name
- $d_j$: demand for customer $j$, from column "demand" in table_id file_1_view_0

All parameters are mapped directly from the CSV files as described above.