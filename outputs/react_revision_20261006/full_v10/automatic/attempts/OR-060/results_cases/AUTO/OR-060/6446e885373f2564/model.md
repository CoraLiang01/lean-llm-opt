##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Supermarket demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$
2. Supplier activation logic:
   $$
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   $$
   where $M_i = \sum_{j \in J} d_j$ is a valid upper bound for each supplier (since no explicit supplier capacity is given).
3. Variable domains:
   $$
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   $$

##### Index Sets and Data Mapping

- $I$: set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 (fixed_cost.csv)
- $J$: set of supermarkets, from column "customer" in table_id file_0_view_0 (demand.csv)
- $d_j$: demand of supermarket $j$, from column "demand" in table_id file_0_view_0 (demand.csv)
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0 (fixed_cost.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$, from table_id file_2_view_0 (transportation_costs.csv), with rows indexed by "Unnamed: 0" (supplier) and columns by supermarket IDs ("C1", ..., "C12")
- $M_i$: upper bound for supplier $i$, set to $\sum_{j \in J} d_j$ (sum of all supermarket demands), derived from demand.csv

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 0"
- $J$: file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand"
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier), column $j$ (supermarket)
- $M_i$: $\sum_{j \in J} d_j$, with $d_j$ from file_0_view_0, column "demand"