#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of vehicle types (from products.csv, column ProductName)
- $J$: set of warehouses (from capacity.csv, column Warehouse ID)

**Parameters:**
- $v_i$: value (benefit coefficient) of vehicle type $i \in I$ (from products.csv, column Value)
- $w_i$: weight (storage requirement) of vehicle type $i \in I$ (from products.csv, column Weight)
- $C_j$: capacity of warehouse $j \in J$ (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_{ij}$: number of vehicles of type $i \in I$ to store in warehouse $j \in J$, integer, $x_{ij} \geq 0$

**Objective:**
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

**Constraints:**
1. **Warehouse Capacity Constraints:**
   \[
   \sum_{i \in I} w_i \cdot x_{ij} \leq C_j, \quad \forall j \in J
   \]
2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- **products.csv**
  - Vehicle type identifiers: ProductName
  - Benefit coefficients: Value
  - Storage requirement per unit: Weight
- **capacity.csv**
  - Warehouse identifiers: Warehouse ID
  - Capacity limits: Capacity

Each parameter and index set is mapped to its exact table and column as above.