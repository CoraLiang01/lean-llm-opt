### Mathematical Model

**Index Sets:**
- $I$: set of storage areas (indexed by $i$), from file_0_view_0, column StorageID
- $J$: set of air conditioner types (indexed by $j$), from file_1_view_0, column ProductName

**Parameters:**
- $c_i$: capacity of storage area $i$ (file_0_view_0, column Capacity, key StorageID)
- $v_j$: value of air conditioner type $j$ (file_1_view_0, column Value, key ProductName)
- $w_j$: size (weight) of air conditioner type $j$ (file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{ij}$: number of units of air conditioner type $j$ placed in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
- Storage area capacity:
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
- Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

### Data Mapping

- $I$: file_0_view_0, column StorageID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, key StorageID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: integer variable for each $(i,j)$ pair

All parameters and index sets are defined directly from the returned CSV data.