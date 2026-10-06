**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $v_p$: value per unit of product $p$ (from Value in file_1_view_0)
- $w_p$: weight per unit of product $p$ (from Weight in file_1_view_0)
- $C_s$: capacity (weight limit) of shelf $s$ (from Capacity in file_0_view_0)

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

**Subject to:**

- **Shelf Capacity Constraints:**
  \[
  \sum_{p \in P} w_p \, x_{s,p} \leq C_s \qquad \forall s \in S
  \]

- **Nonnegativity and Integrality:**
  \[
  x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
  \]

---

**Data Mapping**

- $S$: All ShelfID in `file_0_view_0`, column `ShelfID`
- $P$: All ProductName in `file_1_view_0`, column `ProductName`
- $v_p$: `file_1_view_0`, column `Value`, keyed by `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, keyed by `ProductName`
- $C_s$: `file_0_view_0`, column `Capacity`, keyed by `ShelfID`
- $x_{s,p}$: Decision variable for each $(s,p) \in S \times P$