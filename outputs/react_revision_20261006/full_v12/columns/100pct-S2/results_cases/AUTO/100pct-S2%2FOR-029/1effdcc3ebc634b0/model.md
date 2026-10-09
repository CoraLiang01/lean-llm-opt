### Mathematical Model

**Sets:**
- $I$: set of displays (indexed by $i$), with $i \in$ all ShelfID in file_0_view_0.
- $J$: set of products (indexed by $j$), with $j \in$ all ProductName in file_1_view_0.

**Parameters:**
- $c_i$: capacity of display $i$ (file_0_view_0, column Capacity, key ShelfID).
- $v_j$: value per unit of product $j$ (file_1_view_0, column Value, key ProductName).
- $w_j$: weight per unit of product $j$ (file_1_view_0, column Weight, key ProductName).

**Decision Variables:**
- $x_{ij}$: number of units of product $j$ placed on display $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Model:**

Maximize total value:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

Subject to display capacity constraints:
$$
\sum_{j \in J} w_j x_{ij} \leq c_i, \quad \forall i \in I
$$

Minimum allocation of the first product (the product in the first row of file_1_view_0, denoted $j^*$):
$$
\sum_{i \in I} x_{i j^*} \geq 5
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
$$

---

### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID.
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName.
- $c_i$: file_0_view_0, column Capacity, key ShelfID.
- $v_j$: file_1_view_0, column Value, key ProductName.
- $w_j$: file_1_view_0, column Weight, key ProductName.
- $j^*$: ProductName in source_row 0 of file_1_view_0 (products.csv).
- $x_{ij}$: Number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer).