**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of bookshelves, indexed by $s$ (from file_0_view_0, column BookshelfID)
- $B$: Set of books, indexed by $b$ (from file_1_view_0, column ProductName)

**Parameters:**
- $C_s$: Capacity of bookshelf $s$ (from file_0_view_0, column Capacity)
- $v_b$: Value of book $b$ (from file_1_view_0, column Value)
- $w_b$: Weight of book $b$ (from file_1_view_0, column Weight)

**Decision Variables:**
- $x_{sb} \in \mathbb{Z}_{\geq 0}$: Number of units of book $b$ placed on bookshelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{b \in B} v_b \cdot x_{sb}
\]

**Constraints:**

1. **Bookshelf Capacity Constraints:**  
   For each bookshelf $s \in S$,
   \[
   \sum_{b \in B} w_b \cdot x_{sb} \leq C_s
   \]

2. **Nonnegativity and Integrality:**
   \[
   x_{sb} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, b \in B
   \]

---

**Data Mapping**

- $S$ (bookshelves): file_0_view_0, column BookshelfID
- $C_s$: file_0_view_0, column Capacity, keyed by BookshelfID
- $B$ (books): file_1_view_0, column ProductName
- $v_b$: file_1_view_0, column Value, keyed by ProductName
- $w_b$: file_1_view_0, column Weight, keyed by ProductName

**Variables:**
- $x_{sb}$: Number of units of book $b$ on bookshelf $s$ (indexed by BookshelfID and ProductName)