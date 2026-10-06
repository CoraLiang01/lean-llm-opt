**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of storage areas, indexed by $s$ (from all StorageID in file_0_view_0)
- $P$: set of air conditioner types, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: capacity of storage area $s$ (Capacity column in file_0_view_0, indexed by StorageID)
- $v_p$: value per unit of air conditioner type $p$ (Value column in file_1_view_0, indexed by ProductName)
- $w_p$: size (weight) per unit of air conditioner type $p$ (Weight column in file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{sp}$: number of units of air conditioner type $p$ to place in storage area $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**
1. **Storage Area Capacity Constraints:**  
   For each storage area $s \in S$,
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq c_s
   \]
2. **Nonnegativity and Integrality:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$: All StorageID in `file_0_view_0` (capacity.csv)
- $P$: All ProductName in `file_1_view_0` (products.csv)
- $c_s$: `file_0_view_0`, column `Capacity`, key `StorageID`
- $v_p$: `file_1_view_0`, column `Value`, key `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, key `ProductName`