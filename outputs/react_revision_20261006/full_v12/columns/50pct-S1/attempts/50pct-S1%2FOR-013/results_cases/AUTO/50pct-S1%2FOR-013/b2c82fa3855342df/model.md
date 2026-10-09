### Mathematical Model

**Sets:**
- $I$: set of storage areas, indexed by $i$ (from file_0_view_0, column StorageID)
- $J$: set of air conditioner types, indexed by $j$ (from file_1_view_0, column ProductName)

**Parameters:**
- $c_i$: capacity of storage area $i$ (from file_0_view_0, column Capacity)
- $v_j$: value of one unit of air conditioner type $j$ (from file_1_view_0, column Value)
- $w_j$: size (weight) of one unit of air conditioner type $j$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{ij}$: number of units of air conditioner type $j$ placed in storage area $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Storage Area Capacity:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$: file_0_view_0, column StorageID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, columns [StorageID, Capacity]
- $v_j$: file_1_view_0, columns [ProductName, Value]
- $w_j$: file_1_view_0, columns [ProductName, Weight]
- $x_{ij}$: number of units of air conditioner type $j$ in storage area $i$ (decision variable, indexed by StorageID and ProductName)