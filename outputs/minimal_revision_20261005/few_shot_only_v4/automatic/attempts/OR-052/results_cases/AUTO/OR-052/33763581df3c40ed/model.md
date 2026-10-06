**Abstract Mathematical Model**

**Index Sets:**
- $B$: set of bookshelves (indexed by $b$), from capacity.csv, column BookshelfID
- $P$: set of books (indexed by $p$), from products.csv, column ProductName

**Parameters:**
- $C_b$: capacity of bookshelf $b$ (capacity.csv, column Capacity)
- $v_p$: value of book $p$ (products.csv, column Value)
- $w_p$: weight of book $p$ (products.csv, column Weight)

**Decision Variables:**
- $x_{bp} \in \mathbb{Z}_{\geq 0}$: number of units of book $p$ placed on bookshelf $b$

**Objective:**
\[
\max \sum_{b \in B} \sum_{p \in P} v_p \cdot x_{bp}
\]

**Constraints:**
1. **Bookshelf Capacity Constraints:**  
   For each bookshelf $b \in B$,
   \[
   \sum_{p \in P} w_p \cdot x_{bp} \leq C_b
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{bp} \in \mathbb{Z}_{\geq 0} \quad \forall b \in B,\, p \in P
   \]

---

**Data Mapping**

- $B$: capacity.csv, column BookshelfID
- $C_b$: capacity.csv, column Capacity (keyed by BookshelfID)
- $P$: products.csv, column ProductName
- $v_p$: products.csv, column Value (keyed by ProductName)
- $w_p$: products.csv, column Weight (keyed by ProductName)