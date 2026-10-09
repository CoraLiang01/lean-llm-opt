#### Index Sets

- $S$: set of shelves (from file_0_view_0, column ShelfID)
- $P$: set of products (from file_1_view_0, column ProductName)

#### Parameters

- $C_s$: capacity of shelf $s \in S$ (from file_0_view_0, column Capacity)
- $v_p$: value of product $p \in P$ (from file_1_view_0, column Value)
- $w_p$: weight of product $p \in P$ (from file_1_view_0, column Weight)

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

- Integer non-negativity:
  $$
  x_{s,p} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S, \forall p \in P
  $$

---

#### Data Mapping

- $S$, $C_s$: file_0_view_0 (capacity.csv), column ShelfID and Capacity (all records, no filter)
- $P$, $v_p$, $w_p$: file_1_view_0 (products.csv), columns ProductName, Value, Weight (all records, no filter)