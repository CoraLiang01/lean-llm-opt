##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Demand satisfaction for each musician/band:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Warehouses can only ship if activated:
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (the total demand).
3. Variable domains:
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameter Mapping

- $I$: Set of warehouses, from column "Unnamed: 0" in table_id "file_1_view_0" (fixed_cost.csv) and "file_2_view_0" (transportation_costs.csv).
- $J$: Set of musicians/bands, from column "customer" in table_id "file_0_view_0" (demand.csv) and columns "C1", ..., "C7" in "file_2_view_0" (transportation_costs.csv).
- $d_j$: Demand for musician/band $j$, from column "demand" in table_id "file_0_view_0" (demand.csv).
- $f_i$: Fixed cost for warehouse $i$, from column "fixed_costs" in table_id "file_1_view_0" (fixed_cost.csv).
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$, from table_id "file_2_view_0" (transportation_costs.csv), row "Unnamed: 0" = $i$, column $j$.
- $M$: $\sum_{j \in J} d_j$ (total demand), computed from all $d_j$ in "file_0_view_0".

##### Data Mapping

- $I$: All unique values in "Unnamed: 0" of "file_1_view_0" and "file_2_view_0".
- $J$: All unique values in "customer" of "file_0_view_0" and columns "C1"–"C7" in "file_2_view_0".
- $d_j$: "demand" column in "file_0_view_0", indexed by "customer".
- $f_i$: "fixed_costs" column in "file_1_view_0", indexed by "Unnamed: 0".
- $c_{ij}$: "file_2_view_0", row "Unnamed: 0" = $i$, column $j$.
- $M$: $\sum_{j \in J} d_j$ from "file_0_view_0".

All parameters and sets are defined directly from the supplied CSV data. No additional constraints or bounds are imposed beyond those specified above.