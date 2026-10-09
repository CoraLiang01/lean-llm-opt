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

1. **Shelf Capacity Constraints** (for all $s \in S$):
   $$
   \sum_{p \in P} w_p \cdot x_{s,p} \leq C_s
   $$

2. **Integrality Constraints** (for all $s \in S$, $p \in P$):
   $$
   x_{s,p} \in \mathbb{Z}_{\geq 0}
   $$

---

#### Data Mapping

- $S$, $C_s$: file_0_view_0 (capacity.csv), column ShelfID and Capacity, all rows
- $P$, $v_p$, $w_p$: file_1_view_0 (products.csv), columns ProductName, Value, Weight, all rows