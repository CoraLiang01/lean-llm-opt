#### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of storage areas, indexed by $s$ (from StorageID in file_0_view_0)
- $P$: Set of air conditioner types, indexed by $p$ (from ProductName in file_1_view_0)

**Parameters:**
- $C_s$: Capacity of storage area $s$ (Capacity from file_0_view_0, indexed by StorageID)
- $v_p$: Value of air conditioner type $p$ (Value from file_1_view_0, indexed by ProductName)
- $w_p$: Weight (size) of air conditioner type $p$ (Weight from file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of air conditioner type $p$ placed in storage area $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
- Storage area capacity (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s
\]
- Integrality and nonnegativity (for all $s \in S$, $p \in P$):
\[
x_{sp} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $S$ (storage areas): file_0_view_0, column StorageID
- $C_s$: file_0_view_0, columns StorageID, Capacity
- $P$ (air conditioner types): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, columns ProductName, Value
- $w_p$: file_1_view_0, columns ProductName, Weight

All parameters and indices are to be used exactly as returned, preserving source order and identifiers.