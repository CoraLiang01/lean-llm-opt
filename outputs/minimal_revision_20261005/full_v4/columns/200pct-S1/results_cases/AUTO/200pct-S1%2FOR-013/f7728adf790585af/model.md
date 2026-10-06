**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of storage areas, indexed by $s$ (from all StorageID in file_0_view_0)
- $P$: Set of air conditioner product types, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: Capacity of storage area $s$ (from Capacity in file_0_view_0, indexed by StorageID)
- $v_p$: Value of one unit of product $p$ (from Value in file_1_view_0, indexed by ProductName)
- $w_p$: Weight (size) of one unit of product $p$ (from Weight in file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ allocated to storage area $s$

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

2. **Nonnegativity and Integrality:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$: All StorageID in `file_0_view_0` (capacity.csv)
- $P$: All ProductName in `file_1_view_0` (products.csv)
- $C_s$: `file_0_view_0`, column `Capacity`, key `StorageID`
- $v_p$: `file_1_view_0`, column `Value`, key `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, key `ProductName`