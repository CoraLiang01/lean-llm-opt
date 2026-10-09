##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Supermarket demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Supplier activation:
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ is a sufficiently large constant (total demand), ensuring that if $y_i = 0$, then $x_{ij} = 0$ for all $j$.
3. Variable domains:
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}, \quad \forall i \in I,\, j \in J
   \]

##### Index Sets

- $I$: set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: set of supermarkets, from column "customer" in table_id file_0_view_0 and columns "C1"..."C12" in file_2_view_0.

##### Parameters and Data Mapping

- $d_j$: demand of supermarket $j$, from column "demand" in table_id file_0_view_0, indexed by "customer".
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0, indexed by "Unnamed: 0".
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$, from table_id file_2_view_0, row "Unnamed: 0" (supplier), column $j$ (supermarket).
- $M_i$: total demand, $M_i = \sum_{j \in J} d_j$ (query-defined expression).

##### Data Mapping

- Supermarket set $J$: file_0_view_0, column "customer"; file_2_view_0, columns "C1"..."C12"
- Supplier set $I$: file_1_view_0, column "Unnamed: 0"; file_2_view_0, row "Unnamed: 0"
- Demand $d_j$: file_0_view_0, columns "customer", "demand"
- Fixed cost $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier), columns "C1"..."C12" (supermarket)
- $M_i$: query-defined, $M_i = \sum_{j \in J} d_j$ (sum over file_0_view_0, column "demand")