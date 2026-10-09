##### Abstract Mathematical Model

Let  
- $I$ = set of platforms (indexed by $i$), with PlatformID from capacity.csv  
- $J$ = set of game genres (indexed by $j$), with ProductName from products.csv  

Parameters:  
- $c_i$ = capacity (memory limit) of platform $i$  
- $v_j$ = value of one unit of game genre $j$  
- $w_j$ = memory requirement (weight) of one unit of game genre $j$  

Decision variables:  
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ = number of units of game genre $j$ to list on platform $i$  

Objective:  
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:  
- Platform memory capacity constraints:  
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

- Integer nonnegativity:  
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

##### Data Mapping

- $I$ (platforms): all PlatformID in table_id=file_0_view_0, column=PlatformID  
- $J$ (game genres): all ProductName in table_id=file_1_view_0, column=ProductName  
- $c_i$: table_id=file_0_view_0, column=Capacity, keyed by PlatformID  
- $v_j$: table_id=file_1_view_0, column=Value, keyed by ProductName  
- $w_j$: table_id=file_1_view_0, column=Weight, keyed by ProductName  
- $x_{ij}$: integer, for all $(i,j) \in I \times J$