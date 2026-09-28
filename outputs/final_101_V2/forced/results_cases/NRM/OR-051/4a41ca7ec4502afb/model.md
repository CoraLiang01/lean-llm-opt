#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of cabinets (from `capacity.csv`, column `CabinetID`)
- $J$: set of coffee products (from `products.csv`, column `ProductName`)

**Parameters:**
- $C_i$: capacity of cabinet $i \in I$ (from `capacity.csv`, column `Capacity`)
- $v_j$: value per unit of product $j \in J$ (from `products.csv`, column `Value`)
- $w_j$: weight per unit of product $j \in J$ (from `products.csv`, column `Weight`)

**Decision Variables:**
- $x_{ij}$: number of units of product $j \in J$ to place in cabinet $i \in I$, integer, $x_{ij} \geq 0$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Cabinet Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   \]
2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

**Data Mapping:**

- Table `file_0_view_0` (`capacity.csv`):  
  - Index set $I$ from column `CabinetID`
  - Parameter $C_i$ from column `Capacity`
- Table `file_1_view_0` (`products.csv`):  
  - Index set $J$ from column `ProductName`
  - Parameter $v_j$ from column `Value`
  - Parameter $w_j$ from column `Weight`