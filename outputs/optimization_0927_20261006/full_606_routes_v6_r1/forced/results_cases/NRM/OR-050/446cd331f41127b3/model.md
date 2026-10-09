#### Index Sets

- $I$: set of displays (indexed by $i$), from `capacity.csv` column `ShelfID`
- $J$: set of products (indexed by $j$), from `products.csv` column `ProductName`

#### Parameters

- $C_i$: capacity of display $i$, from `capacity.csv` column `Capacity`
- $v_j$: value of product $j$, from `products.csv` column `Value`
- $w_j$: weight of product $j$, from `products.csv` column `Weight$

#### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

#### Objective

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

1. **Display Capacity Constraints** (for all $i \in I$):

   $$
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i
   $$

2. **Minimum Placement of First Product** (let $j^*$ be the first product in source order from `products.csv`):

   $$
   \sum_{i \in I} x_{i j^*} \geq 5
   $$

3. **Nonnegativity and Integrality** (for all $i \in I$, $j \in J$):

   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}
   $$

---

#### Data Mapping

- $I$ (displays), $C_i$: from `capacity.csv` (`file_0_view_0`), columns `ShelfID`, `Capacity`
- $J$ (products), $v_j$, $w_j$: from `products.csv` (`file_1_view_0`), columns `ProductName`, `Value`, `Weight`
- The "first product" is the first record in source order from `products.csv` (`file_1_view_0`).

No additional filters or relationships were applied; all records from both files are included as per FALLBACK_FULL_DATA.