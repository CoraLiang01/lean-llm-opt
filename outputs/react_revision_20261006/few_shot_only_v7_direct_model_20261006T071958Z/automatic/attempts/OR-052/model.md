##### Mathematical Model

Let  
- $I$ = set of bookshelves (indexed by $i$), from the column `BookshelfID` in `capacity.csv`
- $J$ = set of books (indexed by $j$), from the column `ProductName` in `products.csv`

Parameters:  
- $c_i$ = capacity of bookshelf $i$ (`Capacity` in `capacity.csv`)
- $v_j$ = value of book $j$ (`Value` in `products.csv`)
- $w_j$ = weight of book $j$ (`Weight` in `products.csv`)

Decision variables:  
- $x_{ij}$ = number of units of book $j$ placed on bookshelf $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

Objective:  
$\displaystyle \max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}$

Subject to:  
$\displaystyle \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I$  
$x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J$

---

##### Data Mapping

- $I$: All `BookshelfID` in `file_0_view_0` (`capacity.csv`)
- $J$: All `ProductName` in `file_1_view_0` (`products.csv`)
- $c_i$: `Capacity` from `file_0_view_0` (`capacity.csv`), keyed by `BookshelfID`
- $v_j$: `Value` from `file_1_view_0` (`products.csv`), keyed by `ProductName`
- $w_j$: `Weight` from `file_1_view_0` (`products.csv`), keyed by `ProductName`
- $x_{ij}$: integer variable for each $(i, j) \in I \times J$