#### Index Sets
- $I$: set of platforms (from capacity.csv, column PlatformID)
- $J$: set of games (from products.csv, column ProductName)

#### Parameters
- $C_i$: memory capacity of platform $i \in I$ (capacity.csv, column Capacity)
- $v_j$: value of game $j \in J$ (products.csv, column Value)
- $w_j$: memory requirement of game $j \in J$ (products.csv, column Weight)

#### Decision Variables
- $x_{ij}$: integer number of units of game $j$ to be listed on platform $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

#### Objective
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

#### Constraints

- Platform memory capacity:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]

- Integer nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- Table: capacity.csv (table_id: file_0_view_0)
    - Platform set $I$: column PlatformID
    - Platform capacity $C_i$: column Capacity
- Table: products.csv (table_id: file_1_view_0)
    - Game set $J$: column ProductName
    - Game value $v_j$: column Value
    - Game memory requirement $w_j$: column Weight

All rows and columns from both tables are included as per the query. No additional filters or restrictions are applied.