##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $J$: set of stores, from column "Store (j)" in Stores_Demands.csv (table_id: file_1_view_0)
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in Stores_Demands.csv (table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from entry (row $i$, column $j$) in TransportationCost.csv (table_id: file_2_view_0, with warehouse and store mapping as per relationships)

##### Data Mapping

- Warehouses $I$ and their parameters $f_i$, $u_i$:  
  - Table: PotentialWarehouses_Costs.csv (table_id: file_0_view_0)  
    - "Warehouse (i)" $\rightarrow$ $i$  
    - "Opening Cost (fi)" $\rightarrow$ $f_i$  
    - "Capacity (units)" $\rightarrow$ $u_i$
- Stores $J$ and their demands $d_j$:  
  - Table: Stores_Demands.csv (table_id: file_1_view_0)  
    - "Store (j)" $\rightarrow$ $j$  
    - "Demand (units, dj)" $\rightarrow$ $d_j$
- Transportation costs $c_{ij}$:  
  - Table: TransportationCost.csv (table_id: file_2_view_0)  
    - Row: warehouse $i$ (from "Unnamed: 1" with mapping to "Warehouse (i)")  
    - Column: store $j$ (from header "W1", "W2", ..., mapped to "Store (j)")  
    - $c_{ij}$ is the entry at (row $i$, column $j$)

All index sets and parameters are defined by the full set of entities in the respective columns of the source tables.