##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each musician/band $j$ receives exactly their demand $d_j$.)

2. **Warehouse activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   (No goods can be shipped from inactive warehouses. $M$ is a sufficiently large constant, e.g., $M = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of warehouses, from column "Unnamed: 0" in table_id "file_1_view_0" and "file_2_view_0".
- $J$: Set of musicians/bands, from column "customer" in table_id "file_0_view_0" and columns "C1"–"C7" in "file_2_view_0".
- $d_j$: Demand for musician/band $j$, from column "demand" in table_id "file_0_view_0".
- $f_i$: Fixed cost for warehouse $i$, from column "fixed_costs" in table_id "file_1_view_0".
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$, from table_id "file_2_view_0", row "Unnamed: 0" = $i$, column $j$.
- $M$: $M = \sum_{j \in J} d_j$ (sum of all demands, using "demand" in table_id "file_0_view_0").

##### Data Mapping

- $I$ = all values in "Unnamed: 0" of "file_1_view_0" and "file_2_view_0"
- $J$ = all values in "customer" of "file_0_view_0" and columns "C1"–"C7" of "file_2_view_0"
- $d_j$ = "demand" in "file_0_view_0"
- $f_i$ = "fixed_costs" in "file_1_view_0"
- $c_{ij}$ = entry in "file_2_view_0" at row "Unnamed: 0" = $i$, column $j$
- $M$ = $\sum_{j \in J} d_j$ from "file_0_view_0"

No additional capacity or proportion constraints are imposed beyond those above.