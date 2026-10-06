**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of storage areas, indexed by $s$ (from all StorageID in file_0_view_0)
- $P$: set of air conditioner types, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: capacity of storage area $s$ (from file_0_view_0, column Capacity, key StorageID)
- $v_p$: value of one unit of product $p$ (from file_1_view_0, column Value, key ProductName)
- $w_p$: size (weight) of one unit of product $p$ (from file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{sp}$: number of units of product $p$ to place in storage area $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**
1. **Storage Area Capacity:**
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All StorageID from file_0_view_0 (capacity.csv)
- $P$: All ProductName from file_1_view_0 (products.csv)
- $C_s$: file_0_view_0, column Capacity, key StorageID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName

All variables, parameters, and constraints are indexed and mapped exactly as above. No data or entity is omitted.