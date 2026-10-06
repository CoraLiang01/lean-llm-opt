##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- $I$: Set of warehouses, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: Set of musicians/bands, from column "customer" in table_id file_0_view_0 and columns "C1"–"C7" in file_2_view_0.
- $d_j$: Demand of musician/band $j$, from column "demand" in table_id file_0_view_0.
- $f_i$: Fixed cost for warehouse $i$, from column "fixed_costs" in table_id file_1_view_0.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$, from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

Let $M = \sum_{j \in J} d_j$ (total demand), which serves as a valid upper bound for any warehouse's total shipments.

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
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: All values in "Unnamed: 0" column of file_1_view_0 and file_2_view_0.
- $J$: All values in "customer" column of file_0_view_0 and columns "C1"–"C7" of file_2_view_0.
- $d_j$: "demand" column in file_0_view_0, indexed by "customer".
- $f_i$: "fixed_costs" column in file_1_view_0, indexed by "Unnamed: 0".
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$.
- $M$: $\sum_{j \in J} d_j$, with $d_j$ as above.