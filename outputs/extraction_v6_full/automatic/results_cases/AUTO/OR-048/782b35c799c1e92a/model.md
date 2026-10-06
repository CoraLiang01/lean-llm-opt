#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of storage areas (indexed by $i$), from capacity.csv, column StorageID.
- $J$: Set of air conditioner types (indexed by $j$), from products.csv, column ProductName.

**Parameters:**
- $c_i$: Capacity of storage area $i$ (from capacity.csv, column Capacity).
- $v_j$: Value of air conditioner type $j$ (from products.csv, column Value).
- $w_j$: Size (weight) of air conditioner type $j$ (from products.csv, column Weight).

**Decision Variables:**
- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$.
  - Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Storage Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i, \quad \forall i \in I
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (storage areas): file_0_view_0, column StorageID
- $J$ (air conditioner types): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, indexed by StorageID
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName

- $x_{ij}$: Decision variable for allocation of ProductName $j$ to StorageID $i$ (no direct data column; defined by model)