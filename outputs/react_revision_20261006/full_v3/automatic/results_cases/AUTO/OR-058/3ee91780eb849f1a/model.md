##### Decision Variables

$x_{ij} \geq 0$: quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $\sum_{j \in J} x_{ij} \leq M y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0)
- $J$ = set of stores (from column "customer" in table_id file_0_view_0 and columns "C1"..."C6" in file_2_view_0)
- $d_j$ = demand for store $j$ (from column "demand" in table_id file_0_view_0)
- $f_i$ = fixed cost for supplier $i$ (from column "fixed_costs" in table_id file_1_view_0)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$)
- $M = \sum_{j \in J} d_j$ (total demand, a valid upper bound for each supplier if no explicit capacity is given)

##### Data Mapping

- $I$: All values in "Unnamed: 0" of file_1_view_0 and file_2_view_0
- $J$: All values in "customer" of file_0_view_0 and columns "C1"..."C6" of file_2_view_0
- $d_j$: "demand" column in file_0_view_0, indexed by "customer"
- $f_i$: "fixed_costs" column in file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $M$: $\sum_{j \in J} d_j$ (sum over "demand" in file_0_view_0)