**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of storage areas, indexed by $s$ (from all StorageID in file_0_view_0)
- $P$: Set of air conditioner types, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: Capacity of storage area $s$ (from file_0_view_0, column Capacity, keyed by StorageID)
- $v_p$: Value of air conditioner type $p$ (from file_1_view_0, column Value, keyed by ProductName)
- $w_p$: Weight (size) of air conditioner type $p$ (from file_1_view_0, column Weight, keyed by ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of air conditioner type $p$ allocated to storage area $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**

1. **Storage Area Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
   \]

2. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All StorageID from `file_0_view_0` (capacity.csv)
- $P$: All ProductName from `file_1_view_0` (products.csv)
- $c_s$: `file_0_view_0`, column `Capacity`, keyed by `StorageID`
- $v_p$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $x_{sp}$: Decision variable for allocation of product $p$ to storage area $s$