#### Index Sets
- $\mathcal{I}$: Set of bookshelves (BookshelfID from capacity.csv)
- $\mathcal{J}$: Set of books (ProductName from products.csv)

#### Parameters
- $C_i$: Capacity of bookshelf $i \in \mathcal{I}$ (Capacity from capacity.csv)
- $v_j$: Value of book $j \in \mathcal{J}$ (Value from products.csv)
- $w_j$: Weight of book $j \in \mathcal{J}$ (Weight from products.csv)

#### Decision Variables
- $x_{ij}$: Number of units of book $j$ placed on bookshelf $i$, for all $i \in \mathcal{I}$, $j \in \mathcal{J}$

#### Objective
$$
\max \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} v_j \cdot x_{ij}
$$

#### Constraints

1. **Bookshelf Capacity Constraints** (for all $i \in \mathcal{I}$):
   $$
   \sum_{j \in \mathcal{J}} w_j \cdot x_{ij} \leq C_i
   $$

2. **Nonnegativity and Integrality** (for all $i \in \mathcal{I},\ j \in \mathcal{J}$):
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}
   $$

---

#### Data Mapping

- $\mathcal{I}$, $C_i$: Table "capacity.csv", column "BookshelfID" (index set), column "Capacity" (parameter)
- $\mathcal{J}$, $v_j$, $w_j$: Table "products.csv", column "ProductName" (index set), columns "Value" and "Weight" (parameters)

No filters were applied; all rows and columns from both tables are included.