##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Customer demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouse IDs (from column "Warehouse ID" in table_id: file_1_view_0)
- $J$: set of customer IDs (from column "Customer ID" in table_id: file_2_view_0)
- $c_{ij}$: unit transportation cost from warehouse $i$ to customer $j$ (from table_id: file_0_view_0, row "Warehouse ID" = $i$, column $j$)
- $f_i$: fixed opening cost for warehouse $i$ (from column "Fixed_Cost" in table_id: file_1_view_0)
- $\text{cap}_i$: capacity of warehouse $i$ (from column "Capacity" in table_id: file_1_view_0)
- $d_j$: demand of customer $j$ (from column "Demand" in table_id: file_2_view_0)

##### Data Mapping

- $I$: All values in "Warehouse ID" (file_1_view_0)
- $J$: All values in "Customer ID" (file_2_view_0)
- $c_{ij}$: Value at (row "Warehouse ID" = $i$, column $j$) in file_0_view_0
- $f_i$: Value in "Fixed_Cost" for warehouse $i$ in file_1_view_0
- $\text{cap}_i$: Value in "Capacity" for warehouse $i$ in file_1_view_0
- $d_j$: Value in "Demand" for customer $j$ in file_2_view_0