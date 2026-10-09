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
2. Domains:
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets and Data Mapping

- $I$: Set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 (fixed_cost.csv)
- $J$: Set of supermarkets, from column "customer" in table_id file_0_view_0 (demand.csv)
- $d_j$: Demand of supermarket $j$, from column "demand" in table_id file_0_view_0 (demand.csv)
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0 (fixed_cost.csv)
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$, from table_id file_2_view_0 (transportation_costs.csv), with rows indexed by "Unnamed: 0" (supplier) and columns by supermarket IDs ("C1", "C2", ...)

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 0"
- $J$: file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand"
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, rows "Unnamed: 0", columns "C1", "C2", ... (aligned to $I$ and $J$)