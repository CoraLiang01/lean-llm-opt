#### Mathematical Model

Let  
- $I$ = set of shelves, indexed by $i$ (from all ShelfID in file_0_view_0)  
- $J$ = set of products, indexed by $j$ (from all ProductName in file_1_view_0)  

Parameters:  
- $c_i$ = capacity of shelf $i$ (file_0_view_0, column Capacity, key ShelfID)  
- $v_j$ = value of product $j$ (file_1_view_0, column Value, key ProductName)  
- $w_j$ = weight of product $j$ (file_1_view_0, column Weight, key ProductName)  

Decision variables:  
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ = number of units of product $j$ placed on shelf $i$

Objective:  
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:  
- Shelf capacity constraints (for all $i \in I$):  
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i
$$

- Integer and nonnegativity constraints (for all $i \in I$, $j \in J$):  
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

#### Data Mapping

- $I$ (shelves): file_0_view_0, column ShelfID
- $J$ (products): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, key ShelfID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName