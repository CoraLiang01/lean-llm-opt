##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Supermarket demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $\sum_{j \in J} x_{ij} \leq M_i y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ is the set of suppliers (from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0)
- $J$ is the set of supermarkets (from column "customer" in table_id file_0_view_0 and columns in file_2_view_0)
- $d_j$ is the demand of supermarket $j$ (from column "demand" in table_id file_0_view_0)
- $f_i$ is the fixed cost for supplier $i$ (from column "fixed_costs" in table_id file_1_view_0)
- $c_{ij}$ is the transportation cost per unit from supplier $i$ to supermarket $j$ (from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$)
- $M_i = \sum_{j \in J} d_j$ (since there are no explicit supplier capacity limits, $M_i$ is a valid upper bound for each supplier)

##### Data Mapping

- $I$: All values in column "Unnamed: 0" of table_id file_1_view_0 and file_2_view_0
- $J$: All values in column "customer" of table_id file_0_view_0 and columns (except "Unnamed: 0") of table_id file_2_view_0
- $d_j$: column "demand" in table_id file_0_view_0, indexed by "customer"
- $f_i$: column "fixed_costs" in table_id file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $M_i$: $\sum_{j \in J} d_j$ (sum over all "demand" in table_id file_0_view_0)