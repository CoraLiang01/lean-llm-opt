#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of shelves, indexed by $i$ (from capacity.csv, column ShelfID)
- $P$: set of products, indexed by $j$ (from products.csv, column ProductName)

**Parameters:**
- $c_i$: capacity of shelf $i$ (from capacity.csv, column Capacity)
- $v_j$: value per unit of product $j$ (from products.csv, column Value)
- $w_j$: weight (space requirement) per unit of product $j$ (from products.csv, column Weight)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $i \in S$,
   \[
   \sum_{j \in P} w_j \, x_{ij} \leq c_i
   \]
2. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
   \]

---

#### Data Mapping

- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, indexed by ShelfID
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName

All data and index sets are to be used as returned, preserving original file and row order.