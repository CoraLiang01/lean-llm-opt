#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves (from file_0_view_0, column ShelfID)
- $P$: set of products (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: capacity of shelf $s \in S$ (from file_0_view_0, column Capacity)
- $v_p$: value per unit of product $p \in P$ (from file_1_view_0, column Value)
- $w_p$: weight per unit of product $p \in P$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s
   \]

2. **Non-negativity and Integrality:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

#### Data Mapping

- $S$ (shelves): file_0_view_0, column ShelfID
- $C_s$ (shelf capacities): file_0_view_0, column Capacity
- $P$ (products): file_1_view_0, column ProductName
- $v_p$ (product values): file_1_view_0, column Value
- $w_p$ (product weights): file_1_view_0, column Weight

All records from both tables are included (no filters applied).