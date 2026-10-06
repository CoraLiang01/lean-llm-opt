**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of storage areas, indexed by $s$ (from all StorageID in file_0_view_0)
- $P$: set of air conditioner types, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: capacity of storage area $s$ (file_0_view_0, column Capacity, key StorageID)
- $v_p$: value per unit of air conditioner type $p$ (file_1_view_0, column Value, key ProductName)
- $w_p$: size (weight) per unit of air conditioner type $p$ (file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of air conditioner type $p$ allocated to storage area $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Storage Area Capacity:**
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All StorageID in `file_0_view_0` (from `capacity.csv`, column `StorageID`)
- $P$: All ProductName in `file_1_view_0` (from `products.csv`, column `ProductName`)
- $c_s$: `file_0_view_0`, column `Capacity`, key `StorageID`
- $v_p$: `file_1_view_0`, column `Value`, key `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, key `ProductName`
- $x_{sp}$: Decision variable for allocation of product $p$ to storage area $s$