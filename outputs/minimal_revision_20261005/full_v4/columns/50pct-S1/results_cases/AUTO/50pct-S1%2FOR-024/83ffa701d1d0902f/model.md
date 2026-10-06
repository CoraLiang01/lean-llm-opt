##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- $I$: Set of warehouses, from file_1_view_0["Unnamed: 0"].
- $J$: Set of musicians/bands, from file_0_view_0["customer"].
- $d_j$: Demand of musician/band $j$, from file_0_view_0["demand"].
- $f_i$: Fixed cost for warehouse $i$, from file_1_view_0["fixed_costs"].
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$, from file_2_view_0, row $i$ (file_1_view_0["Unnamed: 0"]), column $j$ (file_0_view_0["customer"]).
- $M$: A sufficiently large constant, $M = \sum_{j \in J} d_j$.

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
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I$: file_1_view_0["Unnamed: 0"]
- $J$: file_0_view_0["customer"]
- $d_j$: file_0_view_0["demand"], indexed by $j$
- $f_i$: file_1_view_0["fixed_costs"], indexed by $i$
- $c_{ij}$: file_2_view_0, row $i$ (file_1_view_0["Unnamed: 0"]), column $j$ (file_0_view_0["customer"])
- $M = \sum_{j \in J} d_j$ (sum over file_0_view_0["demand"])