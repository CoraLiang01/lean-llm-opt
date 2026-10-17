#### Index Sets

- $I$: set of potential warehouses (from PotentialWarehouses_Costs.csv, column "Warehouse (i)")
- $J$: set of stores (from Stores_Demands.csv, column "Store (j)")

#### Parameters

- $f_i$: opening cost for warehouse $i \in I$ (PotentialWarehouses_Costs.csv, column "Opening Cost (fi)")
- $C_i$: capacity of warehouse $i \in I$ (PotentialWarehouses_Costs.csv, column "Capacity (units)")
- $d_j$: demand of store $j \in J$ (Stores_Demands.csv, column "Demand (units, dj)")
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (TransportationCost.csv, entry at row for $i$, column for $j$)

#### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise
- $x_{ij} \geq 0$: amount supplied from warehouse $i$ to store $j$

#### Objective

Minimize total cost (opening + transportation):

$$
\min \quad \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand Satisfaction** (each store's demand must be met):

   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$

2. **Warehouse Capacity** (do not exceed capacity if opened):

   $$
   \sum_{j \in J} x_{ij} \leq C_i y_i \quad \forall i \in I
   $$

3. **Variable Domains**:

   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J
   $$

---

#### Data Mapping

- **PotentialWarehouses_Costs.csv** (table_id: file_0_view_0):  
  - $I$ from "Warehouse (i)"
  - $f_i$ from "Opening Cost (fi)"
  - $C_i$ from "Capacity (units)"
- **Stores_Demands.csv** (table_id: file_1_view_0):  
  - $J$ from "Store (j)"
  - $d_j$ from "Demand (units, dj)"
- **TransportationCost.csv** (table_id: file_2_view_0, FALLBACK_FULL_DATA):  
  - $c_{ij}$ from entry at row for warehouse $i$ ("Unnamed: 0" = $i$), column for warehouse $j$ ("Wj"), where $j$ corresponds to store $j$ in $J$.

All records and columns from each file are included as per the query. No additional filtering or aggregation is applied.