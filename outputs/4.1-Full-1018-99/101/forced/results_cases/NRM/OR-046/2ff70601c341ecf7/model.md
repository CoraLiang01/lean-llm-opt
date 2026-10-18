#### Index Sets

- $P$: set of products (from products.csv, column ProductName)

#### Parameters

- $v_p$: value/benefit per unit of product $p \in P$ (from products.csv, column Value)
- $w_p$: weight per unit of product $p \in P$ (from products.csv, column Weight)
- $C$: overall stock capacity (from capacity.csv, column Capacity)

#### Decision Variables

- $x_p$: number of units of product $p \in P$ to order each day; $x_p \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{p \in P} v_p \cdot x_p
$$

#### Constraints

1. **Overall Stock Capacity:**
   $$
   \sum_{p \in P} w_p \cdot x_p \leq C
   $$

2. **Non-negativity and Integrality:**
   $$
   x_p \in \mathbb{Z}_{\geq 0}, \quad \forall p \in P
   $$

---

#### Data Mapping

- $P$, $v_p$, $w_p$: from products.csv, columns ProductName, Value, Weight, table_id file_1_view_0 (all records, no filter)
- $C$: from capacity.csv, column Capacity, table_id file_0_view_0 (all records, no filter)