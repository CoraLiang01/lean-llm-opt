#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of display areas (from capacity.csv, column DisplayID)
- $J$: Set of vessel types (from products.csv, column ProductName)

**Parameters:**
- $C_i$: Capacity of display area $i \in I$ (from capacity.csv, column Capacity)
- $v_j$: Value of vessel type $j \in J$ (from products.csv, column Value)
- $s_j$: Size of vessel type $j \in J$ (from products.csv, column Weight)

**Decision Variables:**
- $x_{ij}$: Number of vessels of type $j$ to be placed in display area $i$; $x_{ij} \in \mathbb{Z}_+, \forall i \in I, j \in J$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Display Area Capacity Constraints:**
   \[
   \sum_{j \in J} s_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   \]
2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_+, \quad \forall i \in I, j \in J
   \]

---

#### Data Mapping

- **Display area index set $I$ and capacity parameter $C_i$**:  
  Source: capacity.csv, columns DisplayID (index), Capacity (parameter); table_id: file_0_view_0

- **Vessel type index set $J$, value $v_j$, and size $s_j$**:  
  Source: products.csv, columns ProductName (index), Value (parameter), Weight (parameter); table_id: file_1_view_0