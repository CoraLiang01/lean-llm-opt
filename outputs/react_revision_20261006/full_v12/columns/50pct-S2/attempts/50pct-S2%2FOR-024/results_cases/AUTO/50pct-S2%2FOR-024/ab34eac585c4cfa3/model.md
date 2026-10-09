##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Demand satisfaction: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j\in J$
2. Activation logic: $\sum_{j\in J} x_{ij} \leq M_i y_i,\quad \forall i\in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I = \{\text{S1}, \text{S2}, \text{S3}\}$ (warehouses, from file_1_view_0 and file_2_view_0)
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$ (musicians/bands, from file_0_view_0 and file_2_view_0)
- $d_j$ is the demand for musician/band $j$ (from file_0_view_0, column "demand")
- $f_i$ is the fixed cost for warehouse $i$ (from file_1_view_0, column "fixed_costs")
- $c_{ij}$ is the transportation cost per unit from warehouse $i$ to musician/band $j$ (from file_2_view_0, columns "C1", "C2", "C3")
- $M_i = \sum_{j\in J} d_j$ (a valid upper bound for each warehouse, since there are no explicit warehouse capacity limits)

##### Data Mapping

- $I$: All values in file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 0"]
- $J$: All values in file_0_view_0["customer"] and file_2_view_0 columns ["C1", "C2", "C3"]
- $d_j$: file_0_view_0, columns "customer", "demand"
- $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- $c_{ij}$: file_2_view_0, rows "Unnamed: 0" (warehouses), columns "C1", "C2", "C3" (musicians/bands)
- $M_i$: $\sum_{j\in J} d_j$ (sum over file_0_view_0["demand"])