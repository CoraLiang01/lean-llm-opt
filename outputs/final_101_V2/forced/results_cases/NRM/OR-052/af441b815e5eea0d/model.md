#### Abstract Mathematical Model

**Index Sets:**
- $\mathcal{I}$: Set of bookshelves (indexed by $i$), from `capacity.csv` column `BookshelfID`
- $\mathcal{J}$: Set of books (indexed by $j$), from `products.csv` column `ProductName`

**Parameters:**
- $C_i$: Capacity of bookshelf $i$, from `capacity.csv` column `Capacity`
- $v_j$: Value of book $j$, from `products.csv` column `Value`
- $w_j$: Weight of book $j$, from `products.csv` column `Weight`

**Decision Variables:**
- $x_{ij}$: Number of units of book $j$ placed on bookshelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each bookshelf $i \in \mathcal{I}$,
   \[
   \sum_{j \in \mathcal{J}} w_j \cdot x_{ij} \leq C_i
   \]
2. **Non-negativity and Integrality:**  
   For all $i \in \mathcal{I}$, $j \in \mathcal{J}$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

---

#### Data Mapping

- Table `file_0_view_0` (`capacity.csv`):  
  - Index set $\mathcal{I}$ from column `BookshelfID`
  - Parameter $C_i$ from column `Capacity`
- Table `file_1_view_0` (`products.csv`):  
  - Index set $\mathcal{J}$ from column `ProductName`
  - Parameter $v_j$ from column `Value`
  - Parameter $w_j$ from column `Weight`