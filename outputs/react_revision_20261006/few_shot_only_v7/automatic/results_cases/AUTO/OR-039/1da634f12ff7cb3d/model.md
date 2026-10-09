##### Abstract Mathematical Model

Let  
- $I$ = set of vehicle types (indexed by $i$), from products.csv  
- $J$ = set of warehouses (indexed by $j$), from capacity.csv  

Parameters:  
- $p_i$ = value (benefit coefficient) of vehicle type $i$  
- $w_i$ = weight (space requirement) of vehicle type $i$  
- $C_j$ = capacity of warehouse $j$  

Decision Variables:  
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of vehicle type $i$ to store in warehouse $j$

Objective:  
$\displaystyle \max \sum_{i \in I} \sum_{j \in J} p_i \, x_{ij}$

Subject to:  
$\displaystyle \sum_{i \in I} w_i \, x_{ij} \leq C_j \quad \forall j \in J$  
$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J$

---

##### Data Mapping

- $I$: All records in products.csv, column ProductName
- $J$: All records in capacity.csv, column Warehouse ID
- $p_i$: products.csv, column Value, keyed by ProductName
- $w_i$: products.csv, column Weight, keyed by ProductName
- $C_j$: capacity.csv, column Capacity, keyed by Warehouse ID
- $x_{ij}$: number of units of vehicle type $i$ in warehouse $j$ (decision variable, integer, nonnegative)