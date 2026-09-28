#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of potential warehouses (indexed by $i$), from `PotentialWarehouses_Costs.csv` column `Warehouse (i)`
- $J$: Set of stores (indexed by $j$), from `Stores_Demands.csv` column `Store (j)$

**Parameters:**
- $f_i$: Opening cost for warehouse $i$ (`PotentialWarehouses_Costs.csv`, column `Opening Cost (fi)`)
- $C_i$: Capacity of warehouse $i$ (`PotentialWarehouses_Costs.csv`, column `Capacity (units)`)
- $d_j$: Demand of store $j$ (`Stores_Demands.csv`, column `Demand (units, dj)`)
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$ (`TransportationCost.csv`, entry at row for warehouse $i$, column for warehouse $j$)

**Decision Variables:**
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise
- $x_{ij} \geq 0$: Amount shipped from warehouse $i$ to store $j$

**Objective:**
\[
\min \left( \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \right)
\]

**Constraints:**
1. **Demand Satisfaction:**
   \[
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   \]
2. **Warehouse Capacity:**
   \[
   \sum_{j \in J} x_{ij} \leq C_i y_i \quad \forall i \in I
   \]
3. **Variable Domains:**
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- **PotentialWarehouses_Costs.csv**
  - Table ID: `file_0_view_0`
  - Columns: `Warehouse (i)`, `Opening Cost (fi)`, `Capacity (units)`
- **Stores_Demands.csv**
  - Table ID: `file_1_view_0`
  - Columns: `Store (j)`, `Demand (units, dj)`
- **TransportationCost.csv**
  - Table ID: `file_2_view_0`
  - Matrix: Row index and column index correspond to warehouse and store identifiers as per the file's structure; entries give $c_{ij}$

---

**All sets, parameters, and constraints are defined symbolically and mapped to their exact source columns and table IDs.**