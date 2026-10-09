##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Supermarket demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Supplier activation (no conditional capacity bounds, as none are specified):
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0"
- $J$: set of supermarkets, from column "customer" in table_id "file_0_view_0"
- $d_j$: demand of supermarket $j$, from column "demand" in table_id "file_0_view_0"
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0"
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$, from table_id "file_2_view_0", with rows indexed by "Unnamed: 0" (supplier) and columns by supermarket IDs

##### Data Mapping

- Supermarket set $J$: table_id "file_0_view_0", column "customer"
- Supplier set $I$: table_id "file_1_view_0", column "Unnamed: 0"
- Demand $d_j$: table_id "file_0_view_0", column "demand"
- Fixed cost $f_i$: table_id "file_1_view_0", column "fixed_costs"
- Transportation cost $c_{ij}$: table_id "file_2_view_0", rows "Unnamed: 0", columns "C1", "C2" (supermarket IDs from $J$)

No additional bounds or constraints are imposed beyond those above.