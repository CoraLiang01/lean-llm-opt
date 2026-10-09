##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Supermarket demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq d_j y_i,\quad \forall i \in I,\, j \in J$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:

- $I$ = set of suppliers (from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0)
- $J$ = set of supermarkets (from column "customer" in table_id file_0_view_0 and columns in file_2_view_0)
- $d_j$ = demand of supermarket $j$ (from column "demand" in table_id file_0_view_0)
- $f_i$ = fixed cost for supplier $i$ (from column "fixed_costs" in table_id file_1_view_0)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to supermarket $j$ (from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$)
- $x_{ij}$ = quantity shipped from supplier $i$ to supermarket $j$
- $y_i$ = binary variable indicating if supplier $i$ is open

##### Data Mapping

- $I$: All values in "Unnamed: 0" of file_1_view_0 and file_2_view_0
- $J$: All values in "customer" of file_0_view_0 and columns (except "Unnamed: 0") of file_2_view_0
- $d_j$: "demand" column in file_0_view_0, indexed by "customer"
- $f_i$: "fixed_costs" column in file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $x_{ij}$, $y_i$: decision variables as defined above

All index sets and parameters are defined by the full current contents of the referenced columns and tables.