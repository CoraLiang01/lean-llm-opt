#### Abstract Mathematical Model

Let:
- $I$ = set of storage areas (indexed by $i$), with business identifier StorageID from capacity.csv
- $J$ = set of air conditioner types (indexed by $j$), with business identifier ProductName from products.csv

Parameters:
- $c_i$ = capacity of storage area $i$ (from capacity.csv, column Capacity)
- $v_j$ = value of air conditioner type $j$ (from products.csv, column Value)
- $s_j$ = size (weight) of air conditioner type $j$ (from products.csv, column Weight)

Decision Variables:
- $x_{ij}$ = number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} s_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $c_i$: capacity of storage area $i$  
  Source: capacity.csv (table_id: file_0_view_0), columns: StorageID (index), Capacity (value)

- $v_j$: value of air conditioner type $j$  
  Source: products.csv (table_id: file_1_view_0), columns: ProductName (index), Value (value)

- $s_j$: size (weight) of air conditioner type $j$  
  Source: products.csv (table_id: file_1_view_0), columns: ProductName (index), Weight (value)

- $x_{ij}$: number of units of air conditioner type $j$ in storage area $i$  
  Indexed by StorageID (from capacity.csv) and ProductName (from products.csv)

---

All index sets, parameters, and variables are defined using the exact business identifiers and columns as returned by CSVQA. No data values are hard-coded; all mappings are explicit.