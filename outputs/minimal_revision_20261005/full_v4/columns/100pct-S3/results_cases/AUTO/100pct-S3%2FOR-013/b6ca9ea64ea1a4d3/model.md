**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of storage areas, indexed by $s$ (from all StorageID in file_0_view_0)
- $P$: Set of air conditioner types, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $v_p$: Value of one unit of air conditioner type $p$ (file_1_view_0, column Value, key ProductName)
- $w_p$: Weight (size) of one unit of air conditioner type $p$ (file_1_view_0, column Weight, key ProductName)
- $C_s$: Capacity of storage area $s$ (file_0_view_0, column Capacity, key StorageID)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of air conditioner type $p$ allocated to storage area $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**
1. **Storage Area Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All StorageID in file_0_view_0 (capacity.csv), column StorageID
- $P$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $v_p$: file_1_view_0 (products.csv), column Value, key ProductName
- $w_p$: file_1_view_0 (products.csv), column Weight, key ProductName
- $C_s$: file_0_view_0 (capacity.csv), column Capacity, key StorageID

All parameters and index sets are defined directly from the returned CSV data, preserving original business identifiers and column names.