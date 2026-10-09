##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i, \quad \forall i \in I
   \]
3. **Activation constraints:**  
   \[
   x_{ij} \leq d_j y_i, \quad \forall i \in I, \forall j \in J
   \]
4. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$ (warehouses, from "Warehouse (i)" in PotentialWarehouses_Costs.csv)
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$ (stores, from "Store (j)" in Stores_Demands.csv)
- $f_i$: opening cost for warehouse $i$, from "Opening Cost (fi)" in PotentialWarehouses_Costs.csv
- $u_i$: capacity of warehouse $i$, from "Capacity (units)" in PotentialWarehouses_Costs.csv
- $d_j$: demand of store $j$, from "Demand (units, dj)" in Stores_Demands.csv
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from TransportationCost.csv, where row "W$i$" and column "W$j$" gives $c_{ij}$

**Source-column mapping:**
- PotentialWarehouses_Costs.csv:  
  - "Warehouse (i)" $\rightarrow$ $i$  
  - "Opening Cost (fi)" $\rightarrow$ $f_i$  
  - "Capacity (units)" $\rightarrow$ $u_i$
- Stores_Demands.csv:  
  - "Store (j)" $\rightarrow$ $j$  
  - "Demand (units, dj)" $\rightarrow$ $d_j$
- TransportationCost.csv:  
  - Row "W$i$" and column "W$j$" $\rightarrow$ $c_{ij}$

All indices and parameters are to be used exactly as in the source files.