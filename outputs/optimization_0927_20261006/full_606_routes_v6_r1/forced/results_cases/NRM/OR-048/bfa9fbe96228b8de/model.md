#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of storage areas (indexed by $i$), from `capacity.csv` column `StorageID`
- $J$: Set of air conditioner types (indexed by $j$), from `products.csv` column `ProductName`

**Parameters:**
- $C_i$: Capacity of storage area $i$ (from `capacity.csv` column `Capacity`)
- $v_j$: Value of one unit of air conditioner type $j$ (from `products.csv` column `Value`)
- $w_j$: Size (weight) of one unit of air conditioner type $j$ (from `products.csv` column `Weight`)

**Decision Variables:**
- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Storage Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
   \]
2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- **Storage Areas and Capacities:**  
  Table: `capacity.csv`  
  Columns: `StorageID` (index set $I$), `Capacity` (parameter $C_i$)

- **Air Conditioner Types, Values, and Sizes:**  
  Table: `products.csv`  
  Columns: `ProductName` (index set $J$), `Value` (parameter $v_j$), `Weight` (parameter $w_j$)