#### Index Sets

- $I$: set of displays (from `capacity.csv`, column `ShelfID`)
- $J$: set of products (from `products.csv`, column `ProductName`)

#### Parameters

- $C_i$: capacity of display $i$ (from `capacity.csv`, column `Capacity`)
- $v_j$: value of product $j$ (from `products.csv`, column `Value`)
- $w_j$: weight of product $j$ (from `products.csv`, column `Weight$)

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

2. **Minimum Placement of First Product** (let $j^*$ be the first product in the source order of `products.csv`):

   $$
   \sum_{i \in I} x_{i j^*} \geq 5
   $$

3. **Nonnegativity and Integrality** (for all $i \in I$, $j \in J$):

   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}
   $$

---

#### Data Mapping

- `capacity.csv`, column `ShelfID`: index set $I$ (displays)
- `capacity.csv`, column `Capacity`: parameter $C_i$
- `products.csv`, column `ProductName`: index set $J$ (products)
- `products.csv`, column `Value`: parameter $v_j$
- `products.csv`, column `Weight`: parameter $w_j$

The minimum placement constraint uses the first product as ordered in `products.csv`. All records and columns from both files are included as parameters and index sets.