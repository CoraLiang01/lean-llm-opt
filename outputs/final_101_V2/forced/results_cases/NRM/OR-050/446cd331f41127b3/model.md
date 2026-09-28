#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of displays (indexed by $i$), corresponding to all unique values in `capacity.csv` column `ShelfID`
- $J$: set of products (indexed by $j$), corresponding to all unique values in `products.csv` column `ProductName$

**Parameters:**
- $C_i$: capacity of display $i$ (`capacity.csv`, column `Capacity`, for each $i \in I$)
- $v_j$: value of product $j$ (`products.csv`, column `Value`, for each $j \in J$)
- $w_j$: weight of product $j$ (`products.csv`, column `Weight`, for each $j \in J$)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$, for all $i \in I$, $j \in J$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**

1. **Display Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq C_i, \quad \forall i \in I
   \]

2. **Minimum Placement for First Product:**
   \[
   \sum_{i \in I} x_{i1} \geq 5
   \]
   (where $j=1$ corresponds to the first product as ordered in `products.csv`)

3. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- **Display set $I$ and capacities $C_i$:**  
  Source: `capacity.csv`, columns `ShelfID` (for $I$), `Capacity` (for $C_i$), table_id: `file_0_view_0`
- **Product set $J$, values $v_j$, and weights $w_j$:**  
  Source: `products.csv`, columns `ProductName` (for $J$), `Value` (for $v_j$), `Weight` (for $w_j$), table_id: `file_1_view_0`
- **First product for constraint 2:**  
  The first row of `products.csv` (as ordered in the file), table_id: `file_1_view_0`, column `ProductName`

---

All sets, parameters, and constraints are defined symbolically and mapped to their exact source columns and table_ids. No literal data values or record counts are included.