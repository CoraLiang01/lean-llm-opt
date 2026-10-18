#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves (indexed by $i$), from `file_0_view_0.ShelfID`
- $P$: set of products (indexed by $j$), from `file_1_view_0.ProductName$

**Parameters:**
- $C_i$: capacity of shelf $i$, from `file_0_view_0.Capacity`
- $v_j$: value of product $j$, from `file_1_view_0.Value`
- $w_j$: weight of product $j$, from `file_1_view_0.Weight`

**Decision Variables:**
- $x_{ij}$: integer number of units of product $j$ placed on shelf $i$, $\forall i \in S, j \in P$

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Constraints:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{j \in P} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in S
   \]

2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S, j \in P
   \]

---

**Data Mapping:**

- $S$ (shelves), $C_i$: from `capacity.csv` as `file_0_view_0`, columns `ShelfID`, `Capacity`
- $P$ (products), $v_j$, $w_j$: from `products.csv` as `file_1_view_0`, columns `ProductName`, `Value`, `Weight`