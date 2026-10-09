##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$ (inactive suppliers cannot ship)
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where:
- $I$ is the set of suppliers (from file_1_view_0, column "Unnamed: 3")
- $J$ is the set of stores (from file_0_view_0, column "Customer")
- $d_j$ is the demand of store $j$ (from file_0_view_0, column "demand")
- $f_i$ is the fixed cost for supplier $i$ (from file_1_view_0, column "fixed_costs")
- $c_{ij}$ is the transportation cost per unit from supplier $i$ to store $j$ (from file_2_view_0, row "Unnamed: 4" = supplier $i$, column = store $j$)
- $D_j$ is the demand of store $j$ (from file_0_view_0, column "demand$)

##### Data Mapping

- Suppliers $I$: file_1_view_0, column "Unnamed: 3"
- Stores $J$: file_0_view_0, column "Customer"
- Demand $d_j$: file_0_view_0, column "demand"
- Fixed cost $f_i$: file_1_view_0, column "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 4" = supplier $i$, column = store $j$ (column names must be mapped to store names)
- $x_{ij}$, $y_i$: as defined above

Note: The mapping between store names in file_0_view_0 ("Customer") and columns in file_2_view_0 must be established for $c_{ij}$. Each $x_{ij}$ is only allowed positive if $y_i = 1$ (supplier open), enforced by constraint 2.