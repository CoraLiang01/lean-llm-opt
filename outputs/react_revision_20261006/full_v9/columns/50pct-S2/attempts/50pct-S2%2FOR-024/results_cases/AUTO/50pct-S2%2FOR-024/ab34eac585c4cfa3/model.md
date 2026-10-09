##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i$ is a sufficiently large constant, e.g., $M_i = \sum_{j \in J} d_j$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of warehouses (from file_1_view_0, column "Unnamed: 0")
- $J$: Set of musicians/bands (from file_0_view_0, column "customer")
- $d_j$: Demand of musician/band $j$ (from file_0_view_0, column "demand")
- $f_i$: Fixed cost for warehouse $i$ (from file_1_view_0, column "fixed_costs")
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)
- $M_i$: Big-M for warehouse $i$ (set as $\sum_{j \in J} d_j$)

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 0"
- $J$: file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand"
- $f_i$: file_1_view_0, column "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $M_i$: $\sum_{j \in J} d_j$ (sum over file_0_view_0, column "demand")