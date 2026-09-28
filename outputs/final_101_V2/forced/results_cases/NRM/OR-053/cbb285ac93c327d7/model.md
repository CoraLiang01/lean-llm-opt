#### Index Sets

- $S$: set of shelves (from capacity.csv, column ShelfID)
- $P$: set of products (from products.csv, column ProductName)

#### Parameters

- $C_s$: capacity of shelf $s \in S$ (from capacity.csv, column Capacity)
- $v_p$: value of product $p \in P$ (from products.csv, column Value)
- $w_p$: weight of product $p \in P$ (from products.csv, column Weight)

#### Decision Variables

- $x_{s,p}$: number of units of product $p$ placed on shelf $s$; $x_{s,p} \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
$$

#### Constraints

1. **Shelf Capacity Constraints** (for all $s \in S$):

$$
\sum_{p \in P} w_p \cdot x_{s,p} \leq C_s
$$

2. **Non-negativity and Integrality** (for all $s \in S$, $p \in P$):

$$
x_{s,p} \in \mathbb{Z}_{\geq 0}
$$

---

#### Data Mapping

- Table: capacity.csv, table_id: file_0_view_0
    - Shelf identifiers: column ShelfID $\rightarrow$ $S$
    - Shelf capacities: column Capacity $\rightarrow$ $C_s$
- Table: products.csv, table_id: file_1_view_0
    - Product identifiers: column ProductName $\rightarrow$ $P$
    - Product values: column Value $\rightarrow$ $v_p$
    - Product weights: column Weight $\rightarrow$ $w_p$