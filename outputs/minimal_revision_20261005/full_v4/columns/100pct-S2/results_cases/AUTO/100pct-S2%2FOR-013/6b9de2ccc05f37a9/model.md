**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of storage areas, indexed by $s$ (from all StorageID in file_0_view_0)
- $P$: set of air conditioner types, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: capacity of storage area $s$ (Capacity from file_0_view_0, indexed by StorageID)
- $v_p$: value of one unit of air conditioner type $p$ (Value from file_1_view_0, indexed by ProductName)
- $w_p$: size (Weight) of one unit of air conditioner type $p$ (Weight from file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{sp}$: number of units of air conditioner type $p$ to place in storage area $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Storage Area Capacity Constraints:**  
   For each storage area $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All StorageID from `file_0_view_0` (capacity.csv)
- $P$: All ProductName from `file_1_view_0` (products.csv)
- $c_s$: `file_0_view_0`, column `Capacity`, indexed by `StorageID`
- $v_p$: `file_1_view_0`, column `Value`, indexed by `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, indexed by `ProductName`
- $x_{sp}$: Decision variable for allocation of product $p$ to storage area $s$