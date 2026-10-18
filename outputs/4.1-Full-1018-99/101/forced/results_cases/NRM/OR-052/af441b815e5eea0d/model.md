#### Index Sets
- $\mathcal{I}$: Set of bookshelves (indexed by $i$), from `file_0_view_0` column `BookshelfID`
- $\mathcal{J}$: Set of books (indexed by $j$), from `file_1_view_0` column `ProductName`

#### Parameters
- $C_i$: Capacity of bookshelf $i$, from `file_0_view_0` column `Capacity`
- $v_j$: Value of book $j$, from `file_1_view_0` column `Value`
- $w_j$: Weight of book $j$, from `file_1_view_0` column `Weight`

#### Decision Variables
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of book $j$ placed on bookshelf $i$

#### Objective
$$
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} v_j \cdot x_{ij}
$$

#### Constraints

1. **Bookshelf Capacity Constraints** (for all $i \in \mathcal{I}$):
   $$
   \sum_{j \in \mathcal{J}} w_j \cdot x_{ij} \leq C_i
   $$

2. **Non-negativity and Integrality** (for all $i \in \mathcal{I},\ j \in \mathcal{J}$):
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}
   $$

---

#### Data Mapping

- $\mathcal{I}$, $C_i$: `capacity.csv` (`file_0_view_0`), columns `BookshelfID`, `Capacity`
- $\mathcal{J}$, $v_j$, $w_j$: `products.csv` (`file_1_view_0`), columns `ProductName`, `Value`, `Weight`