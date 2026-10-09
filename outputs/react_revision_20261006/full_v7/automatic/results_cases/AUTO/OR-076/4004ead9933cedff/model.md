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
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of warehouses (Warehouse ID from file_1_view_0 and file_0_view_0)
- $J$: set of customers (Customer ID from file_2_view_0 and file_0_view_0)

##### Parameters and Data Mapping

- $c_{ij}$: unit transportation cost from warehouse $i$ to customer $j$  
  — Source: file_0_view_0, columns: Warehouse ID (rows), C1...C20 (columns)
- $f_i$: fixed opening cost for warehouse $i$  
  — Source: file_1_view_0, columns: Warehouse ID, Fixed_Cost
- $\text{cap}_i$: capacity of warehouse $i$  
  — Source: file_1_view_0, columns: Warehouse ID, Capacity
- $d_j$: demand of customer $j$  
  — Source: file_2_view_0, columns: Customer ID, Demand

##### Data Mapping

- $I$ = all Warehouse ID in file_1_view_0 and file_0_view_0
- $J$ = all Customer ID in file_2_view_0 and file_0_view_0
- $c_{ij}$: file_0_view_0, row "Warehouse ID" = $i$, column $j$
- $f_i$: file_1_view_0, row "Warehouse ID" = $i$, column "Fixed_Cost"
- $\text{cap}_i$: file_1_view_0, row "Warehouse ID" = $i$, column "Capacity"
- $d_j$: file_2_view_0, row "Customer ID" = $j$, column "Demand"