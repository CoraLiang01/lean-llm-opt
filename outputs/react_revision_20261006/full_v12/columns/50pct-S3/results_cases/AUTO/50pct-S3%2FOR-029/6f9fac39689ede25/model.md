#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with $i$ corresponding to each $\text{ShelfID}$ in file_0_view_0.
- $J$ = set of products (indexed by $j$), with $j$ corresponding to each $\text{ProductName}$ in file_1_view_0.
- $c_i$ = capacity of display $i$ ($\text{Capacity}$ from file_0_view_0).
- $v_j$ = value of product $j$ ($\text{Value}$ from file_1_view_0).
- $w_j$ = weight of product $j$ ($\text{Weight}$ from file_1_view_0).
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable, integer, $\geq 0$).

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. Minimum allocation of the first product (for $j^* =$ first $\text{ProductName}$ in file_1_view_0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: All $\text{ShelfID}$ in file_0_view_0 (capacity.csv)
- $J$: All $\text{ProductName}$ in file_1_view_0 (products.csv)
- $c_i$: $\text{Capacity}$ column in file_0_view_0, indexed by $\text{ShelfID}$
- $v_j$: $\text{Value}$ column in file_1_view_0, indexed by $\text{ProductName}$
- $w_j$: $\text{Weight}$ column in file_1_view_0, indexed by $\text{ProductName}$
- $j^*$: The $\text{ProductName}$ in file_1_view_0 with $\text{source_row} = 0$
- $x_{ij}$: Number of units of product $j$ on display $i$ (decision variable, integer, $\geq 0$)