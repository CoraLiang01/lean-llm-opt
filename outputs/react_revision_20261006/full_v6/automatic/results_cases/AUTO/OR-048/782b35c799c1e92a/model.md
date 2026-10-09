#### Mathematical Optimization Model

**Index Sets:**
- $I$: set of storage areas (indexed by $i$), with StorageID from file_0_view_0.
- $J$: set of air conditioner types (indexed by $j$), with ProductName from file_1_view_0.

**Parameters:**
- $c_i$: capacity of storage area $i$ (Capacity from file_0_view_0, indexed by StorageID).
- $v_j$: value of air conditioner type $j$ (Value from file_1_view_0, indexed by ProductName).
- $w_j$: size (Weight) of air conditioner type $j$ (Weight from file_1_view_0, indexed by ProductName).

**Decision Variables:**
- $x_{ij}$: number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Storage Area Capacity:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$: StorageID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, key StorageID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: number of units of ProductName $j$ in StorageID $i$ (decision variable, integer, $\geq 0$)