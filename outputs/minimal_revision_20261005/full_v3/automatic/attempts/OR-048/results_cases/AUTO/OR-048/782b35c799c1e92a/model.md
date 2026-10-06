**Abstract Mathematical Model**

**Sets:**
- $S$: Set of storage areas, indexed by $s$ (from file_0_view_0, column StorageID)
- $P$: Set of air conditioner types, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: Capacity of storage area $s$ (from file_0_view_0, column Capacity)
- $v_p$: Value of air conditioner type $p$ (from file_1_view_0, column Value)
- $w_p$: Weight (size) of air conditioner type $p$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sp}$: Number of units of air conditioner type $p$ allocated to storage area $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Storage Area Capacity Constraints:**  
   For each storage area $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: file_0_view_0, column StorageID
- $C_s$: file_0_view_0, column Capacity, keyed by StorageID
- $P$: file_1_view_0, column ProductName
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName

All variables, parameters, and constraints are indexed and mapped directly to the supplied data columns and business identifiers.