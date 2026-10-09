##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Branch demand:  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Supplier activation:  
   \[
   \sum_{j\in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j\in J} d_j$.
3. Domains:  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: set of branches, from column "customer" in table_id file_0_view_0 and columns "C1",...,"C5" in file_2_view_0.
- $d_j$: demand of branch $j$, from column "demand" in table_id file_0_view_0.
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0.
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.
- $M = \sum_{j\in J} d_j$, with $d_j$ as above.

##### Data Mapping

- Branch demand: table_id file_0_view_0, columns "customer", "demand".
- Supplier fixed cost: table_id file_1_view_0, columns "Unnamed: 0", "fixed_costs".
- Transportation cost matrix: table_id file_2_view_0, rows "Unnamed: 0" (suppliers), columns "C1", "C2", "C3", "C4", "C5" (branches).
- Index sets $I$ and $J$ as above.