## Abstract Mathematical Model

**Index Sets:**
- $I$: Set of bookshelves, indexed by $i$ (BookshelfID from file_0_view_0)
- $J$: Set of books, indexed by $j$ (ProductName from file_1_view_0)

**Parameters:**
- $c_i$: Capacity of bookshelf $i$ (Capacity from file_0_view_0, indexed by BookshelfID)
- $v_j$: Value of book $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$: Weight of book $j$ (Weight from file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{ij}$: Number of units of book $j$ to place on bookshelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Bookshelf Capacity Constraints:**  
   For each bookshelf $i \in I$,
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

## Data Mapping

- $I$ (Bookshelf index): file_0_view_0.BookshelfID
- $J$ (Book index): file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, indexed by BookshelfID
- $v_j$: file_1_view_0.Value, indexed by ProductName
- $w_j$: file_1_view_0.Weight, indexed by ProductName

- Decision variable $x_{ij}$: Number of units of book $j$ (file_1_view_0.ProductName) to place on bookshelf $i$ (file_0_view_0.BookshelfID)