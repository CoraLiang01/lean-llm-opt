**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of bookshelves, indexed by $s$ (from all BookshelfID in file_0_view_0)
- $B$: set of books, indexed by $b$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: capacity of bookshelf $s$ (from Capacity in file_0_view_0, indexed by BookshelfID)
- $v_b$: value of book $b$ (from Value in file_1_view_0, indexed by ProductName)
- $w_b$: weight of book $b$ (from Weight in file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{sb} \in \mathbb{Z}_{\geq 0}$: number of units of book $b$ placed on bookshelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{b \in B} v_b \cdot x_{sb}
\]

**Constraints:**
- **Bookshelf capacity:** For each bookshelf $s \in S$,
\[
\sum_{b \in B} w_b \cdot x_{sb} \leq C_s
\]
- **Integrality and nonnegativity:** For all $s \in S$, $b \in B$,
\[
x_{sb} \in \mathbb{Z}_{\geq 0}
\]

---

**Data Mapping**

- $S$: All BookshelfID in `file_0_view_0` (capacity.csv)
- $B$: All ProductName in `file_1_view_0` (products.csv)
- $C_s$: `file_0_view_0`, column `Capacity`, indexed by `BookshelfID`
- $v_b$: `file_1_view_0`, column `Value`, indexed by `ProductName`
- $w_b$: `file_1_view_0`, column `Weight`, indexed by `ProductName`
- $x_{sb}$: Decision variable for each $(s, b)$ pair

---

**Summary:**  
Maximize the total value of books allocated to bookshelves, subject to each bookshelf's capacity, using integer allocations per book and shelf. All parameters and index sets are mapped directly to the supplied data columns and IDs.