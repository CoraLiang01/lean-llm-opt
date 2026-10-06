**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of storage areas, indexed by $s$ (from all StorageID in file_0_view_0)
- $P$: Set of air conditioner types, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: Capacity of storage area $s$ (from Capacity in file_0_view_0, indexed by StorageID)
- $v_p$: Value of one unit of air conditioner type $p$ (from Value in file_1_view_0, indexed by ProductName)
- $w_p$: Weight (size) of one unit of air conditioner type $p$ (from Weight in file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: Number of units of air conditioner type $p$ allocated to storage area $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

**Subject to:**

1. **Storage Area Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \cdot x_{s,p} \leq C_s \qquad \forall s \in S
   \]

2. **Nonnegativity and Integrality:**
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All StorageID in `file_0_view_0` (capacity.csv)
- $P$: All ProductName in `file_1_view_0` (products.csv)
- $C_s$: `file_0_view_0`, column `Capacity`, keyed by `StorageID`
- $v_p$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $x_{s,p}$: Decision variable for each $(s,p)$ pair

---

**Summary:**  
Allocate integer units of each air conditioner type to each storage area to maximize total value, subject to each area's capacity. All parameters and index sets are mapped directly from the supplied CSV data using the exact business identifiers.