## Abstract Mathematical Model

**Sets:**
- $S$: set of storage areas, indexed by $s$ (from file_0_view_0, column StorageID)
- $P$: set of air conditioner types, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: capacity of storage area $s$ (from file_0_view_0, column Capacity)
- $v_p$: value of one unit of air conditioner type $p$ (from file_1_view_0, column Value)
- $w_p$: size (weight) of one unit of air conditioner type $p$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sp}$: number of units of air conditioner type $p$ to place in storage area $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Storage Area Capacity:**
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s \qquad \forall s \in S
   \]
2. **Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

## Data Mapping

- $S$ (storage areas): file_0_view_0, column StorageID
- $C_s$: file_0_view_0, column Capacity, keyed by StorageID
- $P$ (air conditioner types): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName

All data is used as returned, preserving file and row order, and all variables are nonnegative integers as required.