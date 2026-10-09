##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is opened.

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
   \sum_{j \in J} x_{ij} \leq \text{cap}_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I = \{$W1, W2, W3, W4, W5, W6, W7, W8, W9, W10$\}$ (from `warehouse.csv` and `cost.csv` "Warehouse ID")
- $J = \{$C1, C2, ..., C20$\}$ (from `demand.csv` and `cost.csv` columns "C1"..."C20")
- $f_i$: fixed opening cost for warehouse $i$ (from `warehouse.csv`, column "Fixed_Cost")
- $\text{cap}_i$: capacity of warehouse $i$ (from `warehouse.csv`, column "Capacity")
- $d_j$: demand of customer $j$ (from `demand.csv`, column "Demand")
- $c_{ij}$: unit transportation cost from warehouse $i$ to customer $j$ (from `cost.csv`, row "Warehouse ID" $i$, column $j$)

**Source-column Data Mapping:**
- Warehouse set $I$: `warehouse.csv` "Warehouse ID", `cost.csv` "Warehouse ID"
- Customer set $J$: `demand.csv` "Customer ID", `cost.csv` columns "C1"..."C20"
- $f_i$: `warehouse.csv` "Fixed_Cost"
- $\text{cap}_i$: `warehouse.csv` "Capacity"
- $d_j$: `demand.csv` "Demand"
- $c_{ij}$: `cost.csv` entry at (warehouse $i$, customer $j$)

All parameters are to be used exactly as provided in the source tables.