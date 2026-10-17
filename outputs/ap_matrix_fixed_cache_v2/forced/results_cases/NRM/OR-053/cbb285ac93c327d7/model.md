#### Index Sets

- $S$: set of shelves (from `file_0_view_0`, column `ShelfID`)
- $P$: set of products (from `file_1_view_0`, column `ProductName`)

#### Parameters

- $C_s$: capacity of shelf $s \in S$ (from `file_0_view_0`, column `Capacity`)
- $v_p$: value of product $p \in P$ (from `file_1_view_0`, column `Value`)
- $w_p$: weight of product $p \in P$ (from `file_1_view_0`, column `Weight`)

#### Decision Variables

- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

#### Objective

$$
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
$$

#### Constraints

- Shelf capacity constraints:
  $$
  \sum_{p \in P} w_p \cdot x_{s,p} \leq C_s, \quad \forall s \in S
  $$

- Integer nonnegativity:
  $$
  x_{s,p} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S,\, p \in P
  $$

---

#### Data Mapping

- $S$, $C_s$: from `capacity.csv` (`file_0_view_0`), columns `ShelfID`, `Capacity`
- $P$, $v_p$, $w_p$: from `products.csv` (`file_1_view_0`), columns `ProductName`, `Value`, `Weight`