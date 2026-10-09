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
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of warehouses (from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0)
- $J$: Set of musicians/bands (from column "customer" in table_id file_0_view_0 and columns in file_2_view_0)

##### Parameters and Data Mapping

- $d_j$: Demand of musician/band $j$ (column "demand" in table_id file_0_view_0, indexed by "customer")
- $f_i$: Fixed cost for warehouse $i$ (column "fixed_costs" in table_id file_1_view_0, indexed by "Unnamed: 0")
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$ (matrix in table_id file_2_view_0, rows indexed by "Unnamed: 0", columns by "C1", "C2", "C3")
- $M = \sum_{j \in J} d_j$ (sum of all demands from file_0_view_0)

##### Data Mapping

- $I$: All values in "Unnamed: 0" from file_1_view_0 and file_2_view_0
- $J$: All values in "customer" from file_0_view_0 and columns "C1", "C2", "C3" in file_2_view_0
- $d_j$: file_0_view_0, columns "customer", "demand"
- $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- $c_{ij}$: file_2_view_0, rows "Unnamed: 0", columns "C1", "C2", "C3"
- $M$: $\sum_{j \in J} d_j$ using file_0_view_0, column "demand"

All index sets and parameters are defined directly from the current CSV data. No additional constraints or limits are imposed beyond those described above.