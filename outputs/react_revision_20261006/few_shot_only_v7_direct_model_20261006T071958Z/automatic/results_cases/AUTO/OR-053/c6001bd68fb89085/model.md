##### Mathematical Model

Let  
- $I$ = set of shelves, indexed by $i$ (from all ShelfID in file_0_view_0)  
- $J$ = set of products, indexed by $j$ (from all ProductName in file_1_view_0)  

Parameters:  
- $c_i$ = capacity of shelf $i$ (from Capacity in file_0_view_0)  
- $v_j$ = value per unit of product $j$ (from Value in file_1_view_0)  
- $w_j$ = weight per unit of product $j$ (from Weight in file_1_view_0)  

Decision variables:  
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$

Objective:  
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:  
For all $i \in I$ (each shelf):
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

##### Data Mapping

- $I$ (shelves): all ShelfID in table_id=file_0_view_0, column=ShelfID
- $J$ (products): all ProductName in table_id=file_1_view_0, column=ProductName
- $c_i$: table_id=file_0_view_0, column=Capacity, keyed by ShelfID
- $v_j$: table_id=file_1_view_0, column=Value, keyed by ProductName
- $w_j$: table_id=file_1_view_0, column=Weight, keyed by ProductName
- $x_{ij}$: integer variable for each $(i,j)$ pair

All indices, parameters, and constraints are defined directly from the supplied data.