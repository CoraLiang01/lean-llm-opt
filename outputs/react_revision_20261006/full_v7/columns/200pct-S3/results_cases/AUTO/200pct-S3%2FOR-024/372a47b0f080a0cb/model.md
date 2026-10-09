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
   where $M_i = \sum_{j \in J} d_j$ (since no explicit warehouse capacity is given).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of warehouses (from column "Unnamed: 0" in file_1_view_0 and file_2_view_0)
- $J$: Set of musicians/bands (from column "customer" in file_0_view_0 and columns "C1", "C2", "C3" in file_2_view_0)
- $d_j$: Demand of musician/band $j$ (from column "demand" in file_0_view_0)
- $f_i$: Fixed cost for warehouse $i$ (from column "fixed_costs" in file_1_view_0)
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$ (from file_2_view_0, row "Unnamed: 0" = $i$, column $j$)
- $M_i$: Big-M parameter for each warehouse $i$ (set to $\sum_{j \in J} d_j$)

##### Data Mapping

- $I$: All values in "Unnamed: 0" column of file_1_view_0 and file_2_view_0
- $J$: All values in "customer" column of file_0_view_0 and columns "C1", "C2", "C3" of file_2_view_0
- $d_j$: file_0_view_0, column "demand", indexed by "customer"
- $f_i$: file_1_view_0, column "fixed_costs", indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$
- $M_i$: $\sum_{j \in J} d_j$ (sum over all "demand" in file_0_view_0)

No warehouse capacity limits are specified, so $M_i$ is set as above. All indices and parameters are mapped directly to the CSV data as described.