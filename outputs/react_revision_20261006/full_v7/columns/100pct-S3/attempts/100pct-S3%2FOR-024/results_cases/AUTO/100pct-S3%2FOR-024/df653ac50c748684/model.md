##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Activation constraint:
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ (since no explicit warehouse capacity is given).
3. Variable domains:
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of warehouses (from column "Unnamed: 0" in file_1_view_0 and file_2_view_0)
- $J$: set of musicians/bands (from column "customer" in file_0_view_0 and columns "C1", "C2", "C3" in file_2_view_0)

##### Parameters and Data Mapping

- $d_j$: demand of musician/band $j$ (from column "demand" in file_0_view_0, indexed by "customer")
- $f_i$: fixed cost for warehouse $i$ (from column "fixed_costs" in file_1_view_0, indexed by "Unnamed: 0")
- $c_{ij}$: transportation cost per unit from warehouse $i$ to musician/band $j$ (from file_2_view_0, rows indexed by "Unnamed: 0", columns by "C1", "C2", "C3")
- $M_i$: big-M for warehouse $i$, set to $\sum_{j \in J} d_j$ (sum of all demands, using file_0_view_0)

##### Data Mapping

- $I$: file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 0"]
- $J$: file_0_view_0["customer"] and file_2_view_0 columns ["C1", "C2", "C3"]
- $d_j$: file_0_view_0["demand"], indexed by "customer"
- $f_i$: file_1_view_0["fixed_costs"], indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, rows "Unnamed: 0", columns "C1", "C2", "C3"
- $M_i$: $\sum_{j \in J} d_j$ (from file_0_view_0["demand"])

All parameters and sets are mapped directly from the provided CSV files as described above.