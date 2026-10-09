#### Index Sets
- $I$: set of potential warehouses (from PotentialWarehouses_Costs.csv, column "Warehouse (i)")
- $J$: set of stores (from Stores_Demands.csv, column "Store (j)")

#### Parameters
- $f_i$: opening cost of warehouse $i \in I$ (PotentialWarehouses_Costs.csv, column "Opening Cost (fi)")
- $C_i$: capacity of warehouse $i \in I$ (PotentialWarehouses_Costs.csv, column "Capacity (units)")
- $d_j$: demand of store $j \in J$ (Stores_Demands.csv, column "Demand (units, dj)")
- $c_{ij}$: transportation cost per unit from warehouse $i \in I$ to store $j \in J$ (TransportationCost.csv, entry at row for warehouse $i$, column for warehouse $j$)

#### Decision Variables
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all $i \in I$
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$

#### Objective
Minimize total cost (opening + transportation):
$$
\min \quad \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction** (each store's demand must be met):
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$

2. **Warehouse Capacity** (do not exceed capacity if opened):
   $$
   \sum_{j \in J} x_{ij} \leq C_i y_i, \quad \forall i \in I
   $$

3. **Variable Domains**:
   $$
   y_i \in \{0,1\}, \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in I, \forall j \in J
   $$

---

#### Data Mapping

- **PotentialWarehouses_Costs.csv**:  
  - Index set $I$ from column "Warehouse (i)"  
  - Parameter $f_i$ from column "Opening Cost (fi)"  
  - Parameter $C_i$ from column "Capacity (units)"

- **Stores_Demands.csv**:  
  - Index set $J$ from column "Store (j)"  
  - Parameter $d_j$ from column "Demand (units, dj)"

- **TransportationCost.csv**:  
  - Parameter $c_{ij}$ from entry at row for warehouse $i$ (row label in "Unnamed: 0"), column for warehouse $j$ (column label "Wj")  
  - The mapping between warehouse/store indices and the matrix rows/columns is direct as per the file's structure.

---

This abstract model uses all data returned by the query, with all index sets, parameters, and constraints mapped to their exact sources and columns.