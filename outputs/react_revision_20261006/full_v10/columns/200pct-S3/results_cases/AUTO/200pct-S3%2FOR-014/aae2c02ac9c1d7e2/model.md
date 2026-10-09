Mathematical Model

Index Sets:
- Let S be the set of shelves, with elements indexed by s and identified by ShelfID from file_0_view_0.
- Let P be the set of products, with elements indexed by p and identified by ProductName from file_1_view_0.

Parameters:
- v_p: Value of product p (file_1_view_0, column Value, key ProductName)
- w_p: Weight of product p (file_1_view_0, column Weight, key ProductName)
- C_s: Capacity of shelf s (file_0_view_0, column Capacity, key ShelfID)

Decision Variables:
- x_{s,p}: Number of units of product p allocated to shelf s; x_{s,p} ∈ ℤ₊ (nonnegative integers), ∀ s ∈ S, p ∈ P

Objective:
Maximize total value allocated across all shelves:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
$$

Subject to:

Shelf capacity constraints (for each shelf s ∈ S):
$$
\sum_{p \in P} w_p \, x_{s,p} \leq C_s \qquad \forall s \in S
$$

Nonnegativity and integrality:
$$
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
$$

Data Mapping

Index Sets:
- S (shelf): file_0_view_0, column ShelfID
- P (product): file_1_view_0, column ProductName

Parameters:
- v_p: file_1_view_0, column Value, key ProductName
- w_p: file_1_view_0, column Weight, key ProductName
- C_s: file_0_view_0, column Capacity, key ShelfID

Decision Variables:
- x_{s,p}: allocation of product p to shelf s (indexed by ShelfID and ProductName)

All parameters and index sets are defined directly from the returned CSV data, using the exact column names and business identifiers.