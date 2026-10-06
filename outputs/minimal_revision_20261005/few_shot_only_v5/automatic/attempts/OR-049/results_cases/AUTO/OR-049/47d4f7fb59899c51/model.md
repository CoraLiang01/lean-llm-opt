**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves (indexed by $s$), from capacity.csv [column: ShelfID]
- $P$: Set of products (indexed by $p$), from products.csv [column: ProductName]

**Parameters:**
- $c_s$: Capacity of shelf $s$ (from capacity.csv [column: Capacity])
- $v_p$: Value of product $p$ (from products.csv [column: Value])
- $w_p$: Weight of product $p$ (from products.csv [column: Weight])

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: capacity.csv [column: ShelfID]
- $P$: products.csv [column: ProductName]
- $c_s$: capacity.csv [column: Capacity], keyed by ShelfID
- $v_p$: products.csv [column: Value], keyed by ProductName
- $w_p$: products.csv [column: Weight], keyed by ProductName