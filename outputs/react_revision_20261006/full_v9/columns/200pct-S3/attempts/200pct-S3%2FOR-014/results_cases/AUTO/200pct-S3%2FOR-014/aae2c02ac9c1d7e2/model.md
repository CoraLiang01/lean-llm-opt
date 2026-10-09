#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from ProductName in file_1_view_0)
- $J$ = set of shelves, indexed by $j$ (from ShelfID in file_0_view_0)

Parameters:
- $v_i$ = value per unit of product $i$ (Value, file_1_view_0)
- $w_i$ = weight per unit of product $i$ (Weight, file_1_view_0)
- $C_j$ = capacity of shelf $j$ (Capacity, file_0_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $i$ to place on shelf $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \, x_{ij}
\]

Subject to:
\[
\sum_{i \in I} w_i \, x_{ij} \leq C_j \qquad \forall j \in J
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (products): file_1_view_0.ProductName
- $J$ (shelves): file_0_view_0.ShelfID
- $v_i$: file_1_view_0.Value (keyed by ProductName)
- $w_i$: file_1_view_0.Weight (keyed by ProductName)
- $C_j$: file_0_view_0.Capacity (keyed by ShelfID)
- $x_{ij}$: allocation variable for product $i$ on shelf $j$ (indexed by file_1_view_0.ProductName and file_0_view_0.ShelfID)

All variables $x_{ij}$ are nonnegative integers. Each shelf's total allocated product weight cannot exceed its Capacity. The objective is to maximize total value across all shelves and products.