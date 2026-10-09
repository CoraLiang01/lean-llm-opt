##### Decision Variables

$x_{ij} \geq 0$: quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).

$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$, where $D_j$ is the demand of store $j$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets

$I = \{$S1, S2, S3, S4, S5, S6$\}$ (suppliers, from file_1_view_0.Unnamed: 0 and file_2_view_0.Unnamed: 0)

$J = \{$C1, C2, C3, C4, C5, C6$\}$ (stores, from file_0_view_0.customer and file_2_view_0 columns)

##### Parameters and Data Mapping

- $d_j$: demand of store $j$ (from file_0_view_0, column "demand", indexed by "customer")
- $f_i$: fixed cost for supplier $i$ (from file_1_view_0, column "fixed_costs", indexed by "Unnamed: 0")
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from file_2_view_0, row "Unnamed: 0" for $i$, column $j$)
- $D_j$: demand of store $j$ (same as $d_j$; used for activation constraint)

##### Data Mapping

- $I$: file_1_view_0.Unnamed: 0 and file_2_view_0.Unnamed: 0
- $J$: file_0_view_0.customer and file_2_view_0 columns [C1, C2, C3, C4, C5, C6]
- $d_j$: file_0_view_0, column "demand", indexed by "customer"
- $f_i$: file_1_view_0, column "fixed_costs", indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" for $i$, column $j$
- $x_{ij}$, $y_i$: decision variables as defined above

All index sets and parameters are derived directly from the current CSV files as described.