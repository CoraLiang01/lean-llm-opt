#### Index Sets

- $I$: set of display areas (from capacity.csv, column DisplayID)
- $J$: set of vessel types (from products.csv, column ProductName)

#### Parameters

- $C_i$: capacity of display area $i \in I$ (from capacity.csv, column Capacity)
- $v_j$: value of vessel type $j \in J$ (from products.csv, column Value)
- $w_j$: size (weight/dimension) of vessel type $j \in J$ (from products.csv, column Weight)

#### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of vessels of type $j$ to place in display area $i$

#### Objective

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

1. **Capacity constraint for each display area:**
   $$
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   $$

2. **Non-negativity and integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I, \forall j \in J
   $$

---

#### Data Mapping

- Table: capacity.csv, table_id: file_0_view_0
  - Display area set $I$ from column DisplayID
  - Capacity parameter $C_i$ from column Capacity

- Table: products.csv, table_id: file_1_view_0
  - Vessel type set $J$ from column ProductName
  - Value parameter $v_j$ from column Value
  - Size parameter $w_j$ from column Weight