#### Index Sets

- $I$: set of platforms (from capacity.csv, column PlatformID)
- $J$: set of games (from products.csv, column ProductName)

#### Parameters

- $C_i$: memory capacity of platform $i \in I$ (capacity.csv, column Capacity)
- $v_j$: value of game $j \in J$ (products.csv, column Value)
- $w_j$: memory requirement of game $j \in J$ (products.csv, column Weight)

#### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of game $j$ to be listed on platform $i$

#### Objective

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

1. **Platform Memory Capacity:**
   $$
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   $$

2. **Integer Variables:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- Table: capacity.csv, table_id: file_0_view_0
  - Platform set $I$ from column PlatformID
  - Platform capacity $C_i$ from column Capacity

- Table: products.csv, table_id: file_1_view_0
  - Game set $J$ from column ProductName
  - Game value $v_j$ from column Value
  - Game memory requirement $w_j$ from column Weight

No additional filters were applied; all rows and columns from both tables are included.