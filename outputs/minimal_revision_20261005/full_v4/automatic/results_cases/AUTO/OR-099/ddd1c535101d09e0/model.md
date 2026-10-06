##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Parameters

- $f_i$: Opening cost for warehouse $i$.
- $u_i$: Capacity of warehouse $i$.
- $d_j$: Demand of store $j$.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$.

##### Index Sets

- $I$: Set of warehouses, $I = \{\text{W1}, \text{W2}, \ldots, \text{W11}\}$.
- $J$: Set of stores, $J = \{1, 2, \ldots, 11\}$.

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
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $f_i$, $u_i$ (warehouse opening cost and capacity):  
  Source: PotentialWarehouses_Costs.csv  
  Table: `file_0_view_0`  
  Columns:  
    - Warehouse index: `"Warehouse (i)"`  
    - Opening cost: `"Opening Cost (fi)"`  
    - Capacity: `"Capacity (units)"`

- $d_j$ (store demand):  
  Source: Stores_Demands.csv  
  Table: `file_1_view_0`  
  Columns:  
    - Store index: `"Store (j)"`  
    - Demand: `"Demand (units, dj)"`

- $c_{ij}$ (transportation cost):  
  Source: TransportationCost.csv  
  Table: `file_2_view_0`  
  Matrix:  
    - Row index: `"Unnamed: 0"` (warehouse, mapped to $i$)  
    - Column index: `"W1"`–`"W11"` (store, mapped to $j$)

- Index sets:  
  - $I$: All `"Warehouse (i)"` in `file_0_view_0` and `"Unnamed: 0"` in `file_2_view_0`  
  - $J$: All `"Store (j)"` in `file_1_view_0` and columns `"W1"`–`"W11"` in `file_2_view_0`

---

**All parameters and sets are defined exactly as in the source CSVs.**