##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods supplied from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- $I$: Set of warehouses (from file_1_view_0, column "Unnamed: 0").
- $J$: Set of musicians/bands (from file_0_view_0, column "customer").
- $d_j$: Demand of musician/band $j$ (from file_0_view_0, column "demand").
- $f_i$: Fixed cost for activating warehouse $i$ (from file_1_view_0, column "fixed_costs").
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each musician/band $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = \sum_{j \in J} d_j$ (total demand; serves as a valid upper bound since there are no explicit warehouse capacity limits).

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All values in file_1_view_0, column "Unnamed: 0"
- $J$: All values in file_0_view_0, column "customer"
- $d_j$: file_0_view_0, columns "customer", "demand"
- $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $M = \sum_{j \in J} d_j$ (sum over file_0_view_0, column "demand")