#### Index Sets

- $S$: set of shelves (from file_0_view_0, column ShelfID)
- $P$: set of products (from file_1_view_0, column ProductName)

#### Parameters

- $C_s$: capacity of shelf $s \in S$ (from file_0_view_0, column Capacity)
- $v_p$: value per unit of product $p \in P$ (from file_1_view_0, column Value)
- $w_p$: weight per unit of product $p \in P$ (from file_1_view_0, column Weight)

#### Decision Variables

- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

#### Objective

$$
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
$$

#### Constraints

1. **Shelf Capacity Constraints** (for all $s \in S$):

$$
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s
$$

2. **Non-negativity and Integrality** (for all $s \in S$, $p \in P$):

$$
x_{sp} \in \mathbb{Z}_{\geq 0}
$$

---

#### Data Mapping

- Shelf set $S$, shelf capacities $C_s$: file_0_view_0 (capacity.csv), columns ShelfID, Capacity
- Product set $P$, product values $v_p$, product weights $w_p$: file_1_view_0 (products.csv), columns ProductName, Value, Weight