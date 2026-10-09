##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
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

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse ID" in table_id="file_1_view_0"
- $J$: set of customers, from column "Customer ID" in table_id="file_2_view_0"
- $d_j$: demand of customer $j$, from column "Demand" in table_id="file_2_view_0"
- $f_i$: fixed cost of warehouse $i$, from column "Fixed_Cost" in table_id="file_1_view_0"
- $\text{cap}_i$: capacity of warehouse $i$, from column "Capacity" in table_id="file_1_view_0"
- $c_{ij}$: transportation cost per unit from warehouse $i$ to customer $j$, from table_id="file_0_view_0", row "Warehouse ID" $i$, column $j$

##### Data Mapping

- Warehouses: table_id="file_1_view_0", column "Warehouse ID"
- Customers: table_id="file_2_view_0", column "Customer ID"
- Demand: table_id="file_2_view_0", column "Demand"
- Fixed cost: table_id="file_1_view_0", column "Fixed_Cost"
- Capacity: table_id="file_1_view_0", column "Capacity"
- Transportation cost: table_id="file_0_view_0", row "Warehouse ID", columns "C1"..."C20" (customer IDs)